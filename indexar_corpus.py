"""
indexar_corpus.py - Indexación del corpus HORECA y persistencia en Chroma (Bloques 24 y 25 / U2.5).

Implementa el flujo completo de indexación reproducible para el proyecto HORECA AI Service:
1. Define un corpus de 10 documentos técnicos reales de la plataforma B2B que conecta
   negocios del sector HORECA con proveedores mayoristas, cada uno con su metadata de fuente e ID.
2. Divide los documentos en chunks mediante RecursiveCharacterTextSplitter.
3. Valida la integridad de los chunks antes de consumir llamadas a la API.
4. Calcula embeddings reales con GoogleGenerativeAIEmbeddings ("models/gemini-embedding-001")
   protegiendo las llamadas contra límites de cuota (429 / RESOURCE_EXHAUSTED).
5. Persiste los vectores en Chroma bajo "vectorstore/" y verifica la reconexión desde disco.
"""

import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PERSIST_DIRECTORY: str = "vectorstore/"
COLLECTION_NAME: str = "corpus_horeca"
EMBEDDING_MODEL: str = "models/gemini-embedding-001"

CORPUS_HORECA: List[str] = [
    "El cliente HTTP de la plataforma B2B HORECA reintenta hasta 3 veces ante un TimeoutError o error 503, con espera exponencial creciente entre intentos.",
    "Los errores de autenticacion (401 o clave API invalida) nunca se reintentan en el conector Gemini: fallan de inmediato lanzando un ValueError explicito.",
    "El costo de cada llamada en HORECA AI Service se calcula con totalTokenCount, promptTokenCount y candidatesTokenCount segun la tarifa por 1000 tokens.",
    "El parser de pedidos B2B transforma solicitudes informales de hoteles, restaurantes y cafeterias en esquemas Pydantic con insumo, cantidad y unidad.",
    "El enrutador condicional clasifica cada mensaje de soporte mayorista en pregunta, queja o sugerencia usando RunnableBranch y una ruta por defecto.",
    "Cada cadena del flujo orquestado HORECA incluye with_fallbacks y un respaldo heuristico para que un fallo del modelo no interrumpa la respuesta.",
    "El agente autonomo B2B consulta disponibilidad en el catalogo mayorista y calcula cotizaciones con IVA del 19% mediante herramientas de riesgo bajo.",
    "Crear o cancelar un borrador de pedido entre un negocio comprador y un proveedor mayorista es una accion reversible clasificada como riesgo medio.",
    "Emitir una orden de compra formal o procesar un pago a proveedor es de riesgo alto y requiere aprobacion humana con esperar_aprobacion en 300 segundos.",
    "Las pruebas unitarias del servicio HORECA usan mocks y embeddings deterministas para que la suite completa de pytest corra en menos de dos segundos.",
]

METADATAS_HORECA: List[Dict[str, str]] = [
    {"fuente": "cliente.py"},
    {"fuente": "src/consumo_api.py"},
    {"fuente": "src/consumo_api.py"},
    {"fuente": "src/order_parser.py"},
    {"fuente": "enrutamiento.py"},
    {"fuente": "orquestacion.py"},
    {"fuente": "agente_con_herramientas.py"},
    {"fuente": "agente_con_herramientas.py"},
    {"fuente": "docs/limites_autonomia_agente.md"},
    {"fuente": "tests/test_order_parser.py"},
]

IDS_HORECA: List[str] = [f"doc{i}" for i in range(len(CORPUS_HORECA))]


def verificar_integridad_chunks(
    chunks: List[Document],
    corpus_original: List[str],
) -> Dict[str, Any]:
    """Verifica que no existan chunks vacíos ni duplicados y compara conteo de caracteres."""
    vacios = [i for i, chunk in enumerate(chunks) if not chunk.page_content.strip()]
    textos = [chunk.page_content for chunk in chunks]
    hay_duplicados = len(textos) != len(set(textos))
    total_chars_chunks = sum(len(chunk.page_content) for chunk in chunks)
    total_chars_corpus = sum(len(doc) for doc in corpus_original)
    return {
        "vacios": vacios,
        "hay_duplicados": hay_duplicados,
        "total_chars_chunks": total_chars_chunks,
        "total_chars_corpus": total_chars_corpus,
    }


def con_reintentos(
    operacion: Callable[[], Any],
    max_intentos: int = 4,
    espera_inicial: float = 5.0,
) -> Any:
    """Reintenta con espera exponencial únicamente ante errores 429 o RESOURCE_EXHAUSTED."""
    espera = espera_inicial
    for intento in range(1, max_intentos + 1):
        try:
            return operacion()
        except Exception as error:
            mensaje = str(error)
            es_limite = "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje
            if not es_limite or intento == max_intentos:
                raise
            print(
                f"Límite de peticiones alcanzado. Reintento {intento} de {max_intentos - 1} "
                f"en {espera} s..."
            )
            time.sleep(espera)
            espera *= 2


def obtener_embeddings(embeddings: Optional[Any] = None) -> Any:
    """Inicializa GoogleGenerativeAIEmbeddings cargando la API key desde .env."""
    if embeddings is not None:
        return embeddings
    load_dotenv(override=True)
    clave_api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not clave_api:
        raise ValueError("Falta GEMINI_API_KEY en el archivo '.env' o variables de entorno.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=clave_api,
    )


def _preparar_chunks_e_ids(
    corpus: List[str],
    metadatas: Optional[List[Dict[str, str]]],
    ids: Optional[List[str]],
    chunk_size: int = 200,
    chunk_overlap: int = 30,
) -> tuple[List[Document], List[str]]:
    """Fragmenta corpus, valida integridad no vacía y resuelve IDs reproducibles."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents(corpus, metadatas=metadatas)
    integridad = verificar_integridad_chunks(chunks, corpus)
    if integridad["vacios"]:
        raise ValueError(f"Se detectaron chunks vacíos en índices: {integridad['vacios']}")
    ids_finales = list(ids) if ids is not None and len(ids) == len(chunks) else [
        f"doc{i}" for i in range(len(chunks))
    ]
    return chunks, ids_finales


def indexar_corpus(
    corpus: List[str] = CORPUS_HORECA,
    persist_directory: str = PERSIST_DIRECTORY,
    chunk_size: int = 200,
    chunk_overlap: int = 30,
    metadatas: Optional[List[Dict[str, str]]] = METADATAS_HORECA,
    ids: Optional[List[str]] = IDS_HORECA,
    embeddings: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Parte el corpus en chunks, valida integridad, calcula embeddings y persiste en Chroma."""
    if os.path.exists(persist_directory):
        SharedSystemClient.clear_system_cache()
        shutil.rmtree(persist_directory)

    chunks, ids_finales = _preparar_chunks_e_ids(
        corpus, metadatas=metadatas, ids=ids, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    modelo_embeddings = obtener_embeddings(embeddings)

    vectorstore = con_reintentos(
        lambda: Chroma.from_documents(
            documents=chunks,
            embedding=modelo_embeddings,
            ids=ids_finales,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    )
    print(f"Indexados {len(chunks)} chunks en {persist_directory}")
    return vectorstore


def cargar_vectorstore(
    persist_directory: str = PERSIST_DIRECTORY,
    embeddings: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Reconecta a un vector store Chroma existente en disco sin recalcular embeddings."""
    modelo_embeddings = obtener_embeddings(embeddings)
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=modelo_embeddings,
    )


def cargar_o_crear_indice(
    corpus: List[str] = CORPUS_HORECA,
    metadatas: Optional[List[Dict[str, str]]] = METADATAS_HORECA,
    ids: Optional[List[str]] = IDS_HORECA,
    persist_directory: str = PERSIST_DIRECTORY,
    embeddings: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> tuple[Chroma, str]:
    """Reutiliza el vector store persistido si ya tiene documentos; solo indexa si está vacío."""
    vectorstore = cargar_vectorstore(
        persist_directory=persist_directory,
        embeddings=embeddings,
        collection_name=collection_name,
    )
    if vectorstore.get()["ids"]:
        return vectorstore, "indice_reutilizado"

    chunks, ids_finales = _preparar_chunks_e_ids(corpus, metadatas=metadatas, ids=ids)
    con_reintentos(lambda: vectorstore.add_documents(chunks, ids=ids_finales))
    return vectorstore, "indice_creado"


def actualizar_indice_incremental(
    nuevos_textos: List[str],
    nuevos_metadatas: List[Dict[str, str]],
    nuevos_ids: List[str],
    persist_directory: str = PERSIST_DIRECTORY,
    embeddings: Optional[Any] = None,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, int]:
    """Agrega al índice persistido únicamente los documentos cuyos IDs aún no existan."""
    if not (len(nuevos_textos) == len(nuevos_metadatas) == len(nuevos_ids)):
        raise ValueError("nuevos_textos, nuevos_metadatas y nuevos_ids deben tener igual longitud.")

    vectorstore = cargar_vectorstore(
        persist_directory=persist_directory,
        embeddings=embeddings,
        collection_name=collection_name,
    )
    existentes = set(vectorstore.get()["ids"])
    docs_pendientes: List[Document] = []
    ids_pendientes: List[str] = []

    for texto, meta, doc_id in zip(nuevos_textos, nuevos_metadatas, nuevos_ids):
        if doc_id in existentes or doc_id in ids_pendientes:
            continue
        if not texto.strip():
            raise ValueError(f"Documento incremental {doc_id!r} está vacío.")
        docs_pendientes.append(Document(page_content=texto, metadata=dict(meta)))
        ids_pendientes.append(doc_id)

    if docs_pendientes:
        con_reintentos(lambda: vectorstore.add_documents(docs_pendientes, ids=ids_pendientes))

    return {
        "agregados": len(docs_pendientes),
        "omitidos_por_duplicado": len(nuevos_ids) - len(ids_pendientes),
        "total_indice": len(vectorstore.get()["ids"]),
    }


if __name__ == "__main__":
    indexar_corpus()
    vectorstore_recargado = cargar_vectorstore()
    conteo = vectorstore_recargado._collection.count()
    ids_guardados = vectorstore_recargado.get()["ids"]
    print(f"Conteo verificado en disco: {conteo}")
    print(f"IDs almacenados: {ids_guardados}")
