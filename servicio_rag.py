"""Servicio RAG modular integrado de Horeca AI Service (Unidad 2.8 - Gate 2 / Bloque S1)."""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage

from grafo_supervision import (
    construir_grafo_supervision,
    iniciar_despliegue,
    nuevo_thread_id,
    resolver,
)
from indexar_corpus import (
    COLLECTION_NAME,
    CORPUS_HORECA,
    IDS_HORECA,
    METADATAS_HORECA,
    PERSIST_DIRECTORY,
    cargar_o_crear_indice,
)
from logging_rag import (
    HandlerDeTrazasRAG,
    RESPUESTA_SIN_CONTEXTO,
    construir_mensajes_seguros,
    extraer_texto_respuesta,
    filtrar_documentos_sospechosos,
    huella,
    logger,
    registrar_consumo,
    trace_id_var,
)
from registro_prompts import construir_registro_horeca
from registro_rag import calcular_costo, medir, obtener_chat_model


INDICE_OFFLINE_DIR = "vectorstore_gate2/"
COLLECTION_OFFLINE = "corpus_horeca_gate2"


def _asegurar_cita_en_respuesta(texto: str, fuentes: list[str]) -> str:
    """Garantiza que la respuesta cite explícitamente al menos una fuente verificada."""
    if not fuentes:
        return texto
    if any(f in texto for f in fuentes):
        return texto
    return f"{texto}\nFuentes consultadas: {', '.join(fuentes)}"


def responder_servicio(
    pregunta: str,
    *,
    usar_api_real: bool = False,
    persist_directory: str | None = None,
    collection_name: str | None = None,
    embeddings: Any | None = None,
    chat_model: Any | None = None,
    k: int = 3,
    corpus: list[str] = CORPUS_HORECA,
    metadatas: list[dict[str, str]] = METADATAS_HORECA,
    ids: list[str] = IDS_HORECA,
) -> dict[str, Any]:
    """Ejecuta el pipeline RAG completo (índice persistido, filtro anti-inyección, citas y logs)."""
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    token = trace_id_var.set(trace_id)
    auditoria: list[dict[str, Any]] = []
    tiempos: dict[str, float] = {}

    def _registrar_evento(nombre: str, nivel: str = "info", **campos: Any) -> None:
        datos = {"evento": nombre, **campos}
        auditoria.append({"trace_id": trace_id, **datos})
        getattr(logger, nivel)(nombre, extra={"datos": datos})

    try:
        _registrar_evento(
            "consulta_recibida",
            pregunta_len=len(pregunta),
            pregunta_huella=huella(pregunta),
        )

        if usar_api_real:
            dir_indice = persist_directory or PERSIST_DIRECTORY
            coleccion = collection_name or COLLECTION_NAME
            emb = embeddings
        else:
            dir_indice = persist_directory or INDICE_OFFLINE_DIR
            coleccion = collection_name or COLLECTION_OFFLINE
            emb = embeddings or DeterministicFakeEmbedding(size=768)

        with medir(tiempos, "indice_y_recuperacion"):
            vectorstore, estado_indice = cargar_o_crear_indice(
                corpus=corpus,
                metadatas=metadatas,
                ids=ids,
                persist_directory=dir_indice,
                embeddings=emb,
                collection_name=coleccion,
            )
            _registrar_evento(estado_indice, indice_dir=dir_indice)

            handler = HandlerDeTrazasRAG()
            config_cb = {"callbacks": [handler], "metadata": {"trace_id": trace_id}}
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            documentos = retriever.invoke(pregunta, config=config_cb)
            _registrar_evento("documentos_recuperados", n=len(documentos))

        limpios = filtrar_documentos_sospechosos(documentos)
        descartados = len(documentos) - len(limpios)
        fuentes = sorted(
            {
                str((d.metadata or {}).get("fuente") or (d.metadata or {}).get("source") or "desconocida")
                for d in limpios
            }
        )

        if not limpios:
            _registrar_evento(
                "sin_contexto_confiable",
                nivel="warning",
                n_recuperados=len(documentos),
                n_descartados=descartados,
            )
            consumo = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            return {
                "trace_id": trace_id,
                "estado_indice": estado_indice,
                "respuesta": RESPUESTA_SIN_CONTEXTO,
                "fuentes": [],
                "consumo": consumo,
                "latencia_ms": {k_t: round(v * 1000, 2) for k_t, v in tiempos.items()},
                "costo_usd": 0.0,
                "docs_descartados": descartados,
                "auditoria": auditoria,
            }

        mensajes = construir_mensajes_seguros(pregunta, limpios)
        with medir(tiempos, "generacion"):
            if usar_api_real or chat_model is not None:
                modelo = obtener_chat_model(chat_model)
                salida_modelo = modelo.invoke(mensajes, config=config_cb)
                if not hasattr(salida_modelo, "usage_metadata"):
                    texto_bruto = extraer_texto_respuesta(salida_modelo)
                    in_tok = sum(len(str(m.content)) // 4 for m in mensajes)
                    out_tok = max(1, len(texto_bruto) // 4)
                    salida_modelo = AIMessage(
                        content=texto_bruto,
                        usage_metadata={
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "total_tokens": in_tok + out_tok,
                        },
                    )
            else:
                resumen_principal = limpios[0].page_content
                respuesta_simulada = (
                    f"Según {fuentes[0]} [1]: {resumen_principal}"
                )
                modelo_offline = FakeListLLM(responses=[respuesta_simulada])
                texto_offline = modelo_offline.invoke(mensajes, config=config_cb)
                in_tok = sum(len(str(m.content)) // 4 for m in mensajes)
                out_tok = max(1, len(str(texto_offline)) // 4)
                salida_modelo = AIMessage(
                    content=str(texto_offline),
                    usage_metadata={
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok,
                    },
                )

        consumo = registrar_consumo(salida_modelo)
        auditoria.append({"trace_id": trace_id, "evento": "consumo_llm", **consumo})
        texto_final = _asegurar_cita_en_respuesta(
            extraer_texto_respuesta(salida_modelo),
            fuentes,
        )
        latencia_ms = {etapa: round(seg * 1000, 2) for etapa, seg in tiempos.items()}
        costo_usd = round(
            calcular_costo(consumo["input_tokens"], consumo["output_tokens"]),
            6,
        )
        _registrar_evento(
            "consulta_completada",
            docs_entrada=len(documentos),
            docs_limpios=len(limpios),
            docs_descartados=descartados,
            latencia_total_ms=round(sum(latencia_ms.values()), 2),
        )
        return {
            "trace_id": trace_id,
            "estado_indice": estado_indice,
            "respuesta": texto_final,
            "fuentes": fuentes,
            "consumo": consumo,
            "latencia_ms": latencia_ms,
            "costo_usd": costo_usd,
            "docs_descartados": descartados,
            "auditoria": auditoria,
        }
    finally:
        trace_id_var.reset(token)


def ejecutar_servicio(
    pregunta: str,
    *,
    usar_api_real: bool = False,
    persist_directory: str | None = None,
    collection_name: str | None = None,
    embeddings: Any | None = None,
    chat_model: Any | None = None,
) -> tuple[str, list[str]]:
    """Interfaz simplificada para evaluación Gate 2: retorna (respuesta_con_fuente, fuentes)."""
    resultado = responder_servicio(
        pregunta,
        usar_api_real=usar_api_real,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=embeddings,
        chat_model=chat_model,
    )
    return resultado["respuesta"], resultado["fuentes"]


def demo_supervision_despliegue() -> dict[str, Any]:
    """Ejecuta el grafo LangGraph HITL real sobre la última versión de prompt HORECA."""
    registro = construir_registro_horeca()
    ultima_version = registro.ultima("rag_proveedores_horeca")
    huella_propuesta = ultima_version["huella"]

    app = construir_grafo_supervision()
    parcial, config = iniciar_despliegue(
        huella_propuesta,
        app=app,
        thread_id=nuevo_thread_id("gate2-horeca"),
    )
    assert "__interrupt__" in parcial, "El grafo debe pausarse en el punto de supervisión"
    estado_final = resolver(app, config, aprobar=True)
    return {
        "huella_propuesta": huella_propuesta,
        "pausado_con_interrupt": parcial["__interrupt__"][0].value,
        "estado_final": estado_final,
    }


def main(usar_api_real: bool = False) -> None:
    pregunta_demo = (
        "¿Qué acciones de alto riesgo del agente B2B requieren aprobación humana y qué espera tienen?"
    )
    modo = "API real de Gemini" if usar_api_real else "modo determinista offline"
    print(f"=== Servicio RAG HORECA AI Service ({modo}) ===")
    resultado = responder_servicio(pregunta_demo, usar_api_real=usar_api_real)
    print("\n--- respuesta completa del servicio ---")
    print(
        json.dumps(
            {
                "trace_id": resultado["trace_id"],
                "estado_indice": resultado["estado_indice"],
                "respuesta": resultado["respuesta"],
                "fuentes": resultado["fuentes"],
                "consumo": resultado["consumo"],
                "latencia_ms": resultado["latencia_ms"],
                "costo_usd": resultado["costo_usd"],
                "docs_descartados": resultado["docs_descartados"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert resultado["fuentes"], "La respuesta debe citar al menos una fuente"

    print("\n--- punto de supervisión humana ---")
    supervision = demo_supervision_despliegue()
    print("pausado, pendiente:", supervision["pausado_con_interrupt"])
    print("estado final:", supervision["estado_final"])
    print("Proceso terminado sin errores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servicio integrado Gate 2 HORECA AI Service")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Ejecuta contra el vector store real (vectorstore/) y la API de Gemini.",
    )
    args = parser.parse_args()
    main(usar_api_real=args.real)
