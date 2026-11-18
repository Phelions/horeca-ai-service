"""
recuperacion.py - Recuperación semántica y prompt dinámico con citas (Bloque 26 / U2.5).

Reconecta al índice Chroma ya persistido en disco (sin recalcular embeddings del corpus),
recupera los k chunks semánticamente más cercanos a una pregunta y construye un prompt
dinámico numerado que exige citar cada fuente mediante índices [N].
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from indexar_corpus import COLLECTION_NAME, PERSIST_DIRECTORY, cargar_vectorstore


PROMPT_TEMPLATE: str = (
    "Responde la pregunta usando SOLO el contexto de abajo. Cita la "
    "fuente de cada afirmacion con su numero entre corchetes, como [1]. Si el contexto no "
    "alcanza para responder, dilo explicitamente en vez de inventar.\n\n"
    "Contexto:\n"
    "{contexto}\n\n"
    "Pregunta: {pregunta}\n\n"
    "Respuesta:"
)


def formatear_contexto(chunks_recuperados: List[Any]) -> Tuple[str, Dict[int, str]]:
    """Arma un bloque de contexto numerado para el prompt y un mapa numero -> fuente."""
    lineas: List[str] = []
    citas: Dict[int, str] = {}
    for indice, doc in enumerate(chunks_recuperados, start=1):
        fuente = doc.metadata.get("fuente", "desconocida")
        lineas.append(f"[{indice}] {doc.page_content}")
        citas[indice] = fuente
    contexto = "\n\n".join(lineas)
    return contexto, citas


def construir_prompt(
    pregunta: str,
    persist_directory: str = PERSIST_DIRECTORY,
    k: int = 3,
    embeddings: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> Tuple[str, Dict[int, str]]:
    """Recupera los k chunks más relevantes y arma el prompt dinámico con contexto numerado."""
    vectorstore = cargar_vectorstore(
        persist_directory=persist_directory,
        embeddings=embeddings,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chunks_recuperados = retriever.invoke(pregunta)
    contexto, citas = formatear_contexto(chunks_recuperados)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=pregunta)
    return prompt, citas


def extraer_citas_de_respuesta(
    respuesta_texto: str,
    citas: Dict[int, str],
) -> Dict[int, str]:
    """Extrae los índices [N] citados en la respuesta y los resuelve contra el diccionario citas."""
    indices = [int(num) for num in re.findall(r"\[(\d+)\]", respuesta_texto)]
    return {
        idx: citas.get(idx, "Cita invalida / no encontrada")
        for idx in indices
    }


if __name__ == "__main__":
    preguntas_prueba = [
        "Que pasa si una llamada al cliente tarda demasiado o da error 401?",
        "Que acciones del agente HORECA requieren aprobacion humana de alto riesgo?",
        "Cual es la temperatura ideal para hornear croissants en Paris?",
    ]
    for pregunta_actual in preguntas_prueba:
        print("=" * 70)
        prompt_generado, mapa_citas = construir_prompt(
            pregunta_actual,
            persist_directory=PERSIST_DIRECTORY,
        )
        print(prompt_generado)
        print("\nMapa de citas:", mapa_citas)
