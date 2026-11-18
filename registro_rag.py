"""
registro_rag.py - Instrumentación de latencia y costo del flujo RAG completo (Bloque 28 Core / U2.5).

Mide por etapa (embed_y_busqueda y generacion) el tiempo real en milisegundos con
time.perf_counter(), registra el consumo real de tokens desde respuesta.usage_metadata,
expone helpers de estimación de costo y persiste cada corrida en registros_latencia_costo.jsonl.
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from evaluacion_recuperacion import casos_evaluacion
from indexar_corpus import COLLECTION_NAME, PERSIST_DIRECTORY, cargar_vectorstore
from recuperacion import PROMPT_TEMPLATE, formatear_contexto


REGISTRO_LATENCIA_COSTO_ARCHIVO: str = "registros_latencia_costo.jsonl"


@contextmanager
def medir(tiempos: Dict[str, float], etapa: str) -> Iterator[None]:
    """Mide la duración en segundos de una etapa utilizando time.perf_counter()."""
    inicio = time.perf_counter()
    yield
    tiempos[etapa] = time.perf_counter() - inicio


def aprox_tokens(texto: str) -> int:
    """Aproximación habitual (~4 caracteres por token) para estimaciones rápidas."""
    return len(texto) // 4


def calcular_costo(
    tokens_entrada: int,
    tokens_salida: int,
    tarifa_entrada_x1000: float = 0.0001,
    tarifa_salida_x1000: float = 0.0004,
) -> float:
    """Calcula el costo total estimado a partir de tokens de entrada y salida."""
    costo_entrada = (tokens_entrada / 1000) * tarifa_entrada_x1000
    costo_salida = (tokens_salida / 1000) * tarifa_salida_x1000
    return costo_entrada + costo_salida


def obtener_chat_model(chat_model: Optional[Any] = None) -> Any:
    """Inicializa ChatGoogleGenerativeAI cargando .env si no se pasó un modelo mock/propio."""
    if chat_model is not None:
        return chat_model
    load_dotenv(override=True)
    clave_api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not clave_api:
        raise ValueError("Falta GEMINI_API_KEY en el archivo '.env' o variables de entorno.")
    modelo = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
    return ChatGoogleGenerativeAI(model=modelo, google_api_key=clave_api)


def _extraer_tokens(usage_metadata: Any) -> Optional[Dict[str, int]]:
    """Extrae el resumen limpio de tokens (input_tokens, output_tokens, total_tokens)."""
    if not usage_metadata:
        return None
    bruto = dict(usage_metadata)
    claves = ("input_tokens", "output_tokens", "total_tokens")
    filtrado = {c: int(bruto[c]) for c in claves if c in bruto}
    return filtrado if filtrado else bruto


def registrar_consulta(
    pregunta: str,
    persist_directory: str = PERSIST_DIRECTORY,
    k: int = 3,
    embeddings: Optional[Any] = None,
    chat_model: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, Any]:
    """Ejecuta una consulta RAG completa midiendo tiempos por etapa y tokens reales."""
    tiempos: Dict[str, float] = {}
    modelo_chat = obtener_chat_model(chat_model)

    with medir(tiempos, "embed_y_busqueda"):
        vectorstore = cargar_vectorstore(
            persist_directory=persist_directory,
            embeddings=embeddings,
            collection_name=collection_name,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        chunks = retriever.invoke(pregunta)

    contexto, _ = formatear_contexto(chunks)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=pregunta)

    with medir(tiempos, "generacion"):
        respuesta = modelo_chat.invoke(prompt)

    registro: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pregunta": pregunta,
        "k": k,
        "tiempos_ms": {etapa: round(s * 1000, 2) for etapa, s in tiempos.items()},
        "tiempo_total_ms": round(sum(tiempos.values()) * 1000, 2),
        "tokens": _extraer_tokens(getattr(respuesta, "usage_metadata", None)),
    }
    return registro


if __name__ == "__main__":
    if os.path.exists(REGISTRO_LATENCIA_COSTO_ARCHIVO):
        os.remove(REGISTRO_LATENCIA_COSTO_ARCHIVO)

    corridas = [
        (casos_evaluacion[0]["pregunta"], 3),
        (casos_evaluacion[2]["pregunta"], 3),
        (casos_evaluacion[8]["pregunta"], 3),
        (casos_evaluacion[0]["pregunta"], 6),
    ]

    with open(REGISTRO_LATENCIA_COSTO_ARCHIVO, "a", encoding="utf-8") as archivo:
        for pregunta_actual, valor_k in corridas:
            registro = registrar_consulta(
                pregunta_actual,
                persist_directory=PERSIST_DIRECTORY,
                k=valor_k,
            )
            print(json.dumps(registro, indent=2, ensure_ascii=False))
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
