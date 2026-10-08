import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai.errors import APIError

# Cargar variables de entorno desde .env
load_dotenv()

class CatalogIndexer:
    """
    Gestiona la vectorización del catálogo mayorista de insumos (HORECA)
    utilizando embeddings de Google Gemini y almacenamiento persistente en ChromaDB.
    """
    def __init__(
        self,
        collection_name: str = "catalogo_horeca",
        persist_directory: str = "chroma_db",
        embedding_model: Optional[str] = None
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró GEMINI_API_KEY en las variables de entorno. "
                "Verifica tu archivo .env"
            )
        
        self.client = genai.Client(api_key=api_key)
        # Modelo oficial activo para vectorización semántica en la API v1beta
        self.embedding_model = embedding_model or os.getenv(
            "GEMINI_EMBEDDING_MODEL", 
            "gemini-embedding-001"
        )
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Inicializar cliente de ChromaDB persistente en disco local
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Similitud coseno para embeddings normalizados
        )

    def generar_embedding(self, texto: str) -> List[float]:
        """
        Genera el vector denso correspondiente a una cadena de texto
        utilizando la API oficial de Google GenAI.
        """
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=texto
            )
            # Extraer los valores vectoriales
            if hasattr(result, "embeddings") and result.embeddings:
                return result.embeddings[0].values
            elif hasattr(result, "embedding") and result.embedding:
                return result.embedding.values
            else:
                raise ValueError("No se obtuvieron valores de embedding en la respuesta de Gemini.")
        except APIError as e:
            raise RuntimeError(f"Error al conectar con la API de Gemini para embeddings: {e}")

    def indexar_catalogo(self, ruta_json: str = "data/catalogo.json", forzar_reindexacion: bool = False) -> int:
        """
        Lee el archivo JSON del catálogo mayorista, calcula los embeddings
        de cada insumo y los persiste en ChromaDB.
        """
        archivo = Path(ruta_json)
        if not archivo.exists():
            raise FileNotFoundError(f"No se encontró el archivo de catálogo en: {ruta_json}")

        # Evitar reindexación innecesaria si ya existen documentos y no se forzó
        conteo_actual = self.collection.count()
        if conteo_actual > 0 and not forzar_reindexacion:
            print(f"La colección '{self.collection_name}' ya contiene {conteo_actual} ítems indexados.")
            return conteo_actual

        if forzar_reindexacion and conteo_actual > 0:
            print(f"Limpiando {conteo_actual} registros existentes para reindexar...")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        with open(archivo, "r", encoding="utf-8") as f:
            productos: List[Dict[str, Any]] = json.load(f)

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        print(f"Generando embeddings para {len(productos)} productos del catálogo...")
        
        for p in productos:
            sku = str(p["sku"])
            nombre = p["nombre"]
            categoria = p["categoria"]
            unidad = p["unidad_minima"]
            precio = float(p["precio_actual"])
            proveedor = p["proveedor"]

            # Texto enriquecido para capturar sinónimos y contexto gastronómico
            documento_semantico = (
                f"Insumo: {nombre}. Categoría: {categoria}. Presentación: {unidad}. "
                f"Proveedor: {proveedor}."
            )

            vector = self.generar_embedding(documento_semantico)

            ids.append(sku)
            embeddings.append(vector)
            documents.append(documento_semantico)
            metadatas.append({
                "sku": sku,
                "nombre": nombre,
                "categoria": categoria,
                "unidad_minima": unidad,
                "precio_actual": precio,
                "proveedor": proveedor
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Indexación finalizada exitosamente: {len(ids)} insumos persistidos en ChromaDB.")
        return len(ids)

    def buscar_producto(self, consulta: str, n_resultados: int = 1) -> List[Dict[str, Any]]:
        """
        Busca semánticamente el insumo del catálogo más afín al término consultado
        (ej: 'posta negra' -> 'Posta Negra Vacuno').
        Retorna la lista de coincidencias con su distancia/score de similitud.
        """
        vector_consulta = self.generar_embedding(consulta)
        
        resultados = self.collection.query(
            query_embeddings=[vector_consulta],
            n_results=n_resultados,
            include=["metadatas", "distances", "documents"]
        )

        coincidencias: List[Dict[str, Any]] = []
        if resultados["ids"] and len(resultados["ids"][0]) > 0:
            for idx in range(len(resultados["ids"][0])):
                # En distancia coseno de ChromaDB, distancia = 1 - cosine_similarity
                # Un valor cercano a 0 representa máxima similitud
                distancia = resultados["distances"][0][idx] if resultados["distances"] else 1.0
                similitud = max(0.0, 1.0 - distancia)
                
                coincidencias.append({
                    "sku": resultados["ids"][0][idx],
                    "metadata": resultados["metadatas"][0][idx] if resultados["metadatas"] else {},
                    "document": resultados["documents"][0][idx] if resultados["documents"] else "",
                    "distancia": round(distancia, 4),
                    "similitud": round(similitud, 4)
                })

        return coincidencias


if __name__ == "__main__":
    print("=" * 60)
    print("Iniciando indexación vectorial del catálogo HORECA...")
    print("=" * 60)

    indexer = CatalogIndexer()
    total_indexados = indexer.indexar_catalogo(ruta_json="data/catalogo.json")

    print("\n" + "=" * 60)
    print("Ejecutando pruebas de búsqueda semántica:")
    print("=" * 60)

    # Pruebas con términos informales de cocina
    terminos_prueba = [
        "posta negra limpia",
        "tomate para ensalada",
        "harina panadera",
        "detergente lavaloza"
    ]

    for termino in terminos_prueba:
        hallazgos = indexer.buscar_producto(termino, n_resultados=1)
        if hallazgos:
            top = hallazgos[0]
            meta = top["metadata"]
            print(f"\n[Consulta]: '{termino}'")
            print(f"  -> Coincidencia: [{meta.get('sku')}] {meta.get('nombre')}")
            print(f"  -> Proveedor:    {meta.get('proveedor')} (${meta.get('precio_actual'):,.0f} / {meta.get('unidad_minima')})")
            print(f"  -> Similitud:    {top['similitud'] * 100:.1f}% (Distancia: {top['distancia']})")
        else:
            print(f"\n[Consulta]: '{termino}' -> Sin resultados.")