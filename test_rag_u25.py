"""
test_rag_u25.py - Suite de pruebas offline para la Unidad 2.5 (Bloques 24 a 28).

Valida la mecánica completa del flujo RAG sin consumir llamadas de red externas:
- Indexación y reconexión en Chroma con DeterministicFakeEmbedding.
- Validación de integridad de chunks y reintentos con backoff ante errores 429.
- Formateo numerado de contexto, construcción del prompt dinámico y extracción de citas.
- Métricas de recuperación (precision@k, recall@k, f1, promediar), casos borde y archivo de resultados.
- Preflight check, cálculo aproximado de tokens/costos, registro de latencia/tokens y lectura JSONL.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from evaluacion_recuperacion import (
    evaluar_recuperacion,
    f1,
    guardar_resultados_evaluacion,
    precision_at_k,
    promediar,
    recall_at_k,
)
from indexar_corpus import (
    CORPUS_HORECA,
    IDS_HORECA,
    METADATAS_HORECA,
    cargar_vectorstore,
    con_reintentos,
    indexar_corpus,
    verificar_integridad_chunks,
)
from leer_registros import resumir_registros
from preflight import preflight_check
from recuperacion import construir_prompt, extraer_citas_de_respuesta, formatear_contexto
from registro_rag import aprox_tokens, calcular_costo, registrar_consulta


def test_indexar_y_recargar_vectorstore_preserva_conteo_e_ids(tmp_path):
    """Verifica indexación, persistencia en tmp_path, metadata fuente e ids explícitos."""
    ruta_vs = str(tmp_path / "vectorstore")
    emb_fake = DeterministicFakeEmbedding(size=16)

    vectorstore = indexar_corpus(
        corpus=CORPUS_HORECA,
        persist_directory=ruta_vs,
        chunk_size=200,
        chunk_overlap=30,
        metadatas=METADATAS_HORECA,
        ids=IDS_HORECA,
        embeddings=emb_fake,
        collection_name="test_horeca",
    )
    assert vectorstore._collection.count() == len(CORPUS_HORECA)

    recargado = cargar_vectorstore(
        persist_directory=ruta_vs,
        embeddings=emb_fake,
        collection_name="test_horeca",
    )
    assert recargado._collection.count() == len(CORPUS_HORECA)
    datos = recargado.get()
    assert set(datos["ids"]) == set(IDS_HORECA)
    assert all(m.get("fuente") for m in datos["metadatas"])


def test_verificar_integridad_chunks_y_reindexado_limpio(tmp_path):
    """Confirma que reindexar limpia el directorio previo y valida integridad sin vacíos ni duplicados."""
    ruta_vs = str(tmp_path / "vs_reindex")
    emb_fake = DeterministicFakeEmbedding(size=16)
    docs = [Document(page_content=txt) for txt in CORPUS_HORECA]

    integridad = verificar_integridad_chunks(docs, CORPUS_HORECA)
    assert integridad["vacios"] == []
    assert integridad["hay_duplicados"] is False
    assert integridad["total_chars_chunks"] == integridad["total_chars_corpus"]

    indexar_corpus(
        corpus=CORPUS_HORECA[:3],
        persist_directory=ruta_vs,
        metadatas=METADATAS_HORECA[:3],
        ids=IDS_HORECA[:3],
        embeddings=emb_fake,
        collection_name="test_reindex",
    )
    vs_segundo = indexar_corpus(
        corpus=CORPUS_HORECA[:3],
        persist_directory=ruta_vs,
        metadatas=METADATAS_HORECA[:3],
        ids=IDS_HORECA[:3],
        embeddings=emb_fake,
        collection_name="test_reindex",
    )
    assert vs_segundo._collection.count() == 3


def test_con_reintentos_reintenta_429_y_propaga_otros_errores(monkeypatch):
    """Verifica backoff ante 429 RESOURCE_EXHAUSTED y fallo inmediato en errores no transitorios."""
    monkeypatch.setattr("indexar_corpus.time.sleep", lambda _: None)

    intentos = {"n": 0}

    def op_429():
        intentos["n"] += 1
        if intentos["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert con_reintentos(op_429, max_intentos=4, espera_inicial=1) == "ok"
    assert intentos["n"] == 3

    def op_auth():
        raise ValueError("API key invalida")

    try:
        con_reintentos(op_auth, max_intentos=3, espera_inicial=1)
        assert False, "Debió propagar ValueError sin reintentar"
    except ValueError as err:
        assert "API key invalida" in str(err)


def test_formatear_contexto_construir_prompt_y_extraer_citas(tmp_path):
    """Valida numeración 1..k, fuentes sin 'desconocida', pregunta íntegra y manejo de cita inválida."""
    ruta_vs = str(tmp_path / "vs_prompt")
    emb_fake = DeterministicFakeEmbedding(size=16)

    indexar_corpus(
        corpus=CORPUS_HORECA,
        persist_directory=ruta_vs,
        metadatas=METADATAS_HORECA,
        ids=IDS_HORECA,
        embeddings=emb_fake,
        collection_name="test_prompt",
    )

    pregunta = "Como se calcula el costo en tokens en HORECA?"
    prompt, citas = construir_prompt(
        pregunta=pregunta,
        persist_directory=ruta_vs,
        k=3,
        embeddings=emb_fake,
        collection_name="test_prompt",
    )

    assert list(citas.keys()) == [1, 2, 3]
    assert all(fuente != "desconocida" for fuente in citas.values())
    assert "[1] " in prompt and "[2] " in prompt and "[3] " in prompt
    assert f"Pregunta: {pregunta}" in prompt
    assert "dilo explicitamente en vez de inventar" in prompt

    resueltas = extraer_citas_de_respuesta("Usa totalTokenCount [1] y algo [99].", citas)
    assert resueltas[1] == citas[1]
    assert resueltas[99] == "Cita invalida / no encontrada"

    texto_fallback, mapa_fallback = formatear_contexto([Document(page_content="Sin meta", metadata={})])
    assert texto_fallback == "[1] Sin meta"
    assert mapa_fallback == {1: "desconocida"}


def test_metricas_recuperacion_casos_borde_y_guardado(tmp_path):
    """Comprueba precision@k, recall@k, F1, trade-off al subir k y formato de resultados_evaluacion.txt."""
    assert precision_at_k([], {"doc0"}) == 0.0
    assert recall_at_k(["doc0"], set()) is None
    assert f1(0.0, 0.0) == 0.0
    assert f1(0.5, None) == 0.0
    assert promediar([1.0, None, 0.5]) == 0.75
    assert promediar([None]) == 0.0

    retriever_mock = MagicMock()
    retriever_mock.invoke.side_effect = [
        [SimpleNamespace(id="doc2"), SimpleNamespace(id="doc0"), SimpleNamespace(id="doc4"), SimpleNamespace(id="doc1")],
        [SimpleNamespace(id="doc1"), SimpleNamespace(id="doc3"), SimpleNamespace(id="doc5"), SimpleNamespace(id="doc2")],
    ]
    casos = [
        {"pregunta": "p1", "ids_relevantes": {"doc0"}},
        {"pregunta": "p2", "ids_relevantes": {"doc9"}},
    ]

    resultados_k3, p3, r3 = evaluar_recuperacion(retriever_mock, casos=casos, k=3)
    assert resultados_k3[0]["precision"] == 1 / 3
    assert resultados_k3[0]["recall"] == 1.0
    assert resultados_k3[1]["precision"] == 0.0
    assert resultados_k3[1]["recall"] == 0.0
    assert abs(p3 - (1 / 6)) < 1e-9
    assert r3 == 0.5

    # Trade-off sobre un ranking fijo al subir k de 3 a 5
    ranking = ["doc2", "doc4", "doc0", "doc1", "doc3"]
    assert recall_at_k(ranking[:5], {"doc0"}) >= recall_at_k(ranking[:3], {"doc0"})
    assert precision_at_k(ranking[:5], {"doc0"}) <= precision_at_k(ranking[:3], {"doc0"})

    archivo_eval = tmp_path / "resultados_evaluacion.txt"
    guardar_resultados_evaluacion(resultados_k3, p3, r3, ruta_archivo=str(archivo_eval), k=3)
    contenido = archivo_eval.read_text(encoding="utf-8")
    assert "Configuracion: k=3, chunk_size=200, chunk_overlap=30" in contenido
    assert f"promedio -> precision: {p3:.3f}  recall: {r3:.3f}" in contenido


def test_preflight_check_ok_y_problemas(tmp_path):
    """Valida preflight cuando vectorstore/.gitignore están correctos y cuando faltan."""
    vs_dir = tmp_path / "vectorstore"
    gitignore = tmp_path / ".gitignore"

    problemas_inicial = preflight_check(str(vs_dir) + "/", str(gitignore))
    assert len(problemas_inicial) == 2

    vs_dir.mkdir()
    gitignore.write_text(".env\n", encoding="utf-8")
    problemas_sin_ignore = preflight_check(str(vs_dir) + "/", str(gitignore))
    assert len(problemas_sin_ignore) == 1

    gitignore.write_text(".env\nvectorstore/\n", encoding="utf-8")
    assert preflight_check(str(vs_dir) + "/", str(gitignore)) == []


def test_registrar_consulta_costos_y_leer_registros(tmp_path):
    """Verifica medición por etapa, conteo de tokens desde usage_metadata, costo y lector JSONL."""
    ruta_vs = str(tmp_path / "vs_reg")
    emb_fake = DeterministicFakeEmbedding(size=16)

    indexar_corpus(
        corpus=CORPUS_HORECA,
        persist_directory=ruta_vs,
        metadatas=METADATAS_HORECA,
        ids=IDS_HORECA,
        embeddings=emb_fake,
        collection_name="test_reg",
    )

    chat_mock = MagicMock()
    chat_mock.invoke.side_effect = [
        SimpleNamespace(content="Respuesta k=3 [1].", usage_metadata={"input_tokens": 120, "output_tokens": 25, "total_tokens": 145}),
        SimpleNamespace(content="Respuesta k=6 [1].", usage_metadata={"input_tokens": 240, "output_tokens": 25, "total_tokens": 265}),
    ]

    reg_k3 = registrar_consulta(
        pregunta=CORPUS_HORECA[0],
        persist_directory=ruta_vs,
        k=3,
        embeddings=emb_fake,
        chat_model=chat_mock,
        collection_name="test_reg",
    )
    reg_k6 = registrar_consulta(
        pregunta=CORPUS_HORECA[0],
        persist_directory=ruta_vs,
        k=6,
        embeddings=emb_fake,
        chat_model=chat_mock,
        collection_name="test_reg",
    )

    assert set(reg_k3.keys()) == {"timestamp", "pregunta", "k", "tiempos_ms", "tiempo_total_ms", "tokens"}
    assert "embed_y_busqueda" in reg_k3["tiempos_ms"]
    assert "generacion" in reg_k3["tiempos_ms"]
    assert reg_k6["tokens"]["input_tokens"] > reg_k3["tokens"]["input_tokens"]

    archivo_jsonl = tmp_path / "registros_latencia_costo.jsonl"
    with open(archivo_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(reg_k3, ensure_ascii=False) + "\n")
        f.write(json.dumps(reg_k6, ensure_ascii=False) + "\n")

    resumen = resumir_registros(str(archivo_jsonl))
    assert len(resumen) == 2
    assert "k=3  input_tokens=120" in resumen[0]
    assert "k=6  input_tokens=240" in resumen[1]

    assert aprox_tokens("12345678") == 2
    assert abs(calcular_costo(1000, 1000, 0.0001, 0.0004) - 0.0005) < 1e-12
