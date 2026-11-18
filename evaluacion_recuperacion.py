"""
evaluacion_recuperacion.py - Evaluación cuantitativa del retriever RAG (Bloque 27 / U2.5).

Evalúa la recuperación semántica del vector store Chroma sobre un conjunto etiquetado de
10 preguntas del dominio HORECA usando Precision@k, Recall@k y F1, guardando un registro
histórico comparable en resultados_evaluacion.txt.
"""

from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from indexar_corpus import COLLECTION_NAME, PERSIST_DIRECTORY, cargar_vectorstore


K: int = 3

casos_evaluacion: List[Dict[str, Any]] = [
    {
        "pregunta": "que hace el cliente si una peticion tarda demasiado o da timeout",
        "ids_relevantes": {"doc0"},
    },
    {
        "pregunta": "que ocurre cuando la clave de api es invalida o devuelve error 401",
        "ids_relevantes": {"doc1"},
    },
    {
        "pregunta": "como se calcula el costo en tokens de cada llamada al modelo",
        "ids_relevantes": {"doc2"},
    },
    {
        "pregunta": "como se extraen y validan los pedidos informales de hoteles y cafeterias",
        "ids_relevantes": {"doc3"},
    },
    {
        "pregunta": "como se clasifican y desvian los mensajes de soporte en queja pregunta o sugerencia",
        "ids_relevantes": {"doc4"},
    },
    {
        "pregunta": "que mecanismo evita que un fallo del modelo detenga el flujo orquestado",
        "ids_relevantes": {"doc5"},
    },
    {
        "pregunta": "que herramientas de bajo riesgo permiten consultar stock y cotizar con IVA",
        "ids_relevantes": {"doc6"},
    },
    {
        "pregunta": "como funcionan los borradores de pedido reversibles de riesgo medio",
        "ids_relevantes": {"doc7"},
    },
    {
        "pregunta": "que acciones de alto riesgo exigen aprobacion humana con limite de tiempo",
        "ids_relevantes": {"doc8"},
    },
    {
        "pregunta": "como se ejecutan las pruebas unitarias rapidas sin gastar cuota de red",
        "ids_relevantes": {"doc9"},
    },
]


def precision_at_k(ids_recuperados: List[str], ids_relevantes: Set[str]) -> float:
    """Fracción de los IDs recuperados que son relevantes (0.0 si no se recuperó ninguno)."""
    if not ids_recuperados:
        return 0.0
    acertados = [id_doc for id_doc in ids_recuperados if id_doc in ids_relevantes]
    return len(acertados) / len(ids_recuperados)


def recall_at_k(ids_recuperados: List[str], ids_relevantes: Set[str]) -> Optional[float]:
    """Fracción de los IDs relevantes efectivamente recuperados (None si no hay relevantes)."""
    if not ids_relevantes:
        return None
    acertados = [id_doc for id_doc in ids_recuperados if id_doc in ids_relevantes]
    return len(acertados) / len(ids_relevantes)


def f1(precision: float, recall: Optional[float]) -> float:
    """Media armónica de precisión y recall."""
    if recall is None or (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def promediar(valores: Iterable[Optional[float]]) -> float:
    """Calcula el promedio ignorando los None y evitando división por cero."""
    validos = [v for v in valores if v is not None]
    return sum(validos) / len(validos) if validos else 0.0


def evaluar_recuperacion(
    retriever: Any,
    casos: List[Dict[str, Any]] = casos_evaluacion,
    k: int = K,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Corre cada caso contra el retriever real y promedia precision@k y recall@k."""
    resultados: List[Dict[str, Any]] = []
    for caso in casos:
        chunks = retriever.invoke(caso["pregunta"])
        ids_recuperados = [c.id for c in chunks][:k]
        p = precision_at_k(ids_recuperados, caso["ids_relevantes"])
        r = recall_at_k(ids_recuperados, caso["ids_relevantes"])
        resultados.append({
            "pregunta": caso["pregunta"],
            "ids_recuperados": ids_recuperados,
            "precision": p,
            "recall": r,
        })
    precision_promedio = promediar(x["precision"] for x in resultados)
    recall_promedio = promediar(x["recall"] for x in resultados)
    return resultados, precision_promedio, recall_promedio


def guardar_resultados_evaluacion(
    resultados: List[Dict[str, Any]],
    precision_promedio: float,
    recall_promedio: float,
    ruta_archivo: str = "resultados_evaluacion.txt",
    k: int = K,
    chunk_size: int = 200,
    chunk_overlap: int = 30,
) -> str:
    """Persiste la evaluación por pregunta y promedios en un archivo de texto comparable."""
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(f"Fecha: {date.today()}\n")
        archivo.write(
            f"Configuracion: k={k}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}\n\n"
        )
        for fila in resultados:
            recall_txt = "n/a" if fila["recall"] is None else f"{fila['recall']:.2f}"
            archivo.write(f"{fila['precision']:.2f}  {recall_txt}  {fila['pregunta']}\n")
        archivo.write(
            f"\npromedio -> precision: {precision_promedio:.3f}  recall: {recall_promedio:.3f}\n"
        )
    return ruta_archivo


if __name__ == "__main__":
    vectorstore = cargar_vectorstore(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    retriever_real = vectorstore.as_retriever(search_kwargs={"k": K})
    resultados_corrida, p_prom, r_prom = evaluar_recuperacion(
        retriever_real,
        casos_evaluacion,
        k=K,
    )
    for fila in resultados_corrida:
        recall_txt = "n/a" if fila["recall"] is None else f"{fila['recall']:.2f}"
        print(f"{fila['precision']:.2f}  {recall_txt}  {fila['pregunta']}")
    print(f"\npromedio -> precision: {p_prom:.3f}  recall: {r_prom:.3f}")

    guardar_resultados_evaluacion(resultados_corrida, p_prom, r_prom)
    print("Resultados guardados en resultados_evaluacion.txt")
