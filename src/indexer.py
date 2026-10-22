"""
================================================================================
MÓDULO: indexer.py
PROPÓSITO: Indexación y Búsqueda Semántica del Catálogo Mayorista (HORECA)
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Permite que el sistema 'entienda' los productos del catálogo de proveedores no solo
por coincidencia exacta de letras, sino por su significado real (búsqueda semántica).
Por ejemplo: si el cocinero pide "posta limpia", el sistema es capaz de entender
que se refiere a "Posta Negra Vacuno", o si pide "tomate ensalada" encuentra
"Tomate Limachino Primera".

¿CON QUIÉN SE COMUNICA?
-----------------------
1. Con Google Gemini API (`client.models.embed_content`): Envía el texto de cada
   producto del catálogo y recibe una lista de números decimales (un 'embedding'
   o vector matemático de 768 dimensiones) que representa el significado del producto.
2. Con ChromaDB (Base de datos vectorial local en la carpeta `chroma_db/`):
   Guarda esos vectores en el disco duro para poder hacer búsquedas instantáneas
   sin tener que recalcularlos cada vez.
3. Con el archivo `data/catalogo.json`: Lee los productos reales de los proveedores
   mayoristas (SKU, nombre, categoría, unidad y precio).

¿CÓMO SE COMUNICA?
------------------
- Utiliza la fórmula de 'Similitud Coseno' (Cosine Similarity): compara el ángulo
  entre el vector de la pregunta y los vectores de los productos guardados.
  Entre más cercano a 1.0 sea el resultado, más parecidos son en significado.
"""

# ==============================================================================
# IMPORTACIÓN DE LIBRERÍAS (¿Para qué sirve cada una?)
# ==============================================================================

# 'os' (Operating System): Nos permite consultar y operar sobre el sistema operativo
# (por ejemplo, comprobar si existen directorios en el disco o leer variables de entorno).
import os

# 'json' (JavaScript Object Notation): Permite leer y escribir archivos de datos estructurados.
# Lo usamos para abrir el catálogo maestro de proveedores ('data/catalogo.json') y convertirlo en objetos Python.
import json

# 'pathlib.Path': Permite manipular rutas de archivos y carpetas de manera moderna y limpia,
# asegurando que el código funcione igual en Windows, Linux o macOS sin problemas de barras inclinadas.
from pathlib import Path

# 'typing': Ayuda a documentar qué tipos de datos reciben y retornan las funciones:
# - List: Una lista ordenada de elementos (ej: una lista de productos encontrados).
# - Dict: Un diccionario de datos con claves y valores (ej: un producto con su precio y categoría).
# - Any: Cualquier tipo de dato cuando puede ser flexible.
# - Optional: Un dato que puede estar presente o ser nulo (None).
from typing import List, Dict, Any, Optional

# 'dotenv': Lee el archivo oculto '.env' en la raíz y carga las contraseñas secretas en el sistema,
# protegiendo tu clave GEMINI_API_KEY para que no quede expuesta.
from dotenv import load_dotenv

# 'chromadb': Es una base de datos vectorial local (corre directo en tu máquina sin instalar servidores externos).
# A diferencia de una base de datos tradicional que busca palabras exactas, ChromaDB guarda los "vectores"
# (coordenadas matemáticas) de los productos y permite encontrar coincidencias por significado semántico.
import chromadb

# 'Settings': Objeto de configuración de ChromaDB para indicarle dónde almacenar los datos físicamente
# en el disco duro y ajustar opciones de persistencia.
from chromadb.config import Settings

# 'google.genai': SDK oficial de Google que conecta tu código con la plataforma Gemini.
# Aquí lo usamos para invocar el modelo de 'embeddings' que convierte textos de productos en listas de números.
from google import genai

# 'APIError': Es el tipo de error específico que envía Google si la petición a su servidor falla
# (por ejemplo si la clave no es válida o se cortó la conexión), permitiéndonos atrapar el error sin que el programa colapse.
from google.genai.errors import APIError

# Cargar las variables del archivo .env al sistema operativo
load_dotenv()


class CatalogIndexer:
    """
    Gestiona la vectorización y búsqueda inteligente del catálogo de insumos gastronómicos.
    """

    def __init__(
        self,
        collection_name: str = "catalogo_horeca",
        persist_directory: str = "chroma_db",
        embedding_model: Optional[str] = None
    ):
        """
        Inicializa las conexiones necesarias:
        - Verifica que exista la clave GEMINI_API_KEY.
        - Conecta con el cliente oficial de Gemini.
        - Abre o crea la base de datos vectorial ChromaDB en la carpeta local.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró GEMINI_API_KEY en las variables de entorno. "
                "Verifica que tu archivo .env contenga una clave válida."
            )
        
        self.client = genai.Client(api_key=api_key)
        
        # Modelo oficial de Google para transformar texto en representaciones vectoriales
        self.embedding_model = embedding_model or os.getenv(
            "GEMINI_EMBEDDING_MODEL", 
            "gemini-embedding-001"
        )
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Conexión persistente con ChromaDB en el disco local
        # Al usar persist_directory, los datos quedan guardados y no se borran al apagar el programa
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Creamos u obtenemos la 'colección' (equivalente a una tabla en bases de datos tradicionales)
        # Usamos distancia coseno ("cosine") porque es el estándar para medir similitud de texto
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def generar_embedding(self, texto: str) -> List[float]:
        """
        PASO 1: CONVERTIR TEXTO EN VECTOR MATEMÁTICO
        
        ¿Qué hace?
        - Envía una frase o descripción a Gemini y le pide que devuelva su vector denso.
        - Por ejemplo, la palabra 'harina' se convierte en una lista de números como [0.021, -0.045, ...]
        
        ¿Para qué sirve?
        - Las computadoras no entienden palabras, pero pueden calcular qué tan cerca
          están dos listas de números entre sí.
        """
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=texto
            )
            # El resultado puede venir en .embeddings o .embedding según la versión del SDK
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
        PASO 2: LEER EL CATÁLOGO Y GUARDARLO EN LA BASE VECTORIAL
        
        ¿Qué hace?
        1. Lee el archivo JSON con los productos del proveedor mayorista.
        2. Revisa si la base de datos ya tiene los productos indexados para no gastar
           llamadas a la API repitiendo el trabajo (a menos que uses forzar_reindexacion=True).
        3. Para cada producto, arma un texto descriptivo enriquecido (nombre, categoría, unidad, proveedor).
        4. Calcula el vector matemático de cada uno y los guarda en ChromaDB junto con sus metadatos.
        
        ¿Qué devuelve?
        - La cantidad total de productos listos y guardados en la base de datos.
        """
        archivo = Path(ruta_json)
        if not archivo.exists():
            raise FileNotFoundError(f"No se encontró el archivo de catálogo en: {ruta_json}")

        conteo_actual = self.collection.count()
        # Si ya hay productos guardados y no pediste reindexar, reutilizamos lo que ya existe
        if conteo_actual > 0 and not forzar_reindexacion:
            print(f"[Aviso]: La colección '{self.collection_name}' ya contiene {conteo_actual} ítems indexados.")
            return conteo_actual

        if forzar_reindexacion and conteo_actual > 0:
            print(f"Limpiando {conteo_actual} registros existentes para reindexar desde cero...")
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

        print(f"Calculando embeddings semánticos para {len(productos)} productos del catálogo...")
        
        for p in productos:
            sku = str(p["sku"])
            nombre = p["nombre"]
            categoria = p["categoria"]
            unidad = p["unidad_minima"]
            precio = float(p["precio_actual"])
            proveedor = p["proveedor"]

            # Texto enriquecido: junta las palabras clave para que la búsqueda sea más inteligente
            documento_semantico = (
                f"Insumo: {nombre}. Categoría: {categoria}. Presentación: {unidad}. "
                f"Proveedor: {proveedor}."
            )

            # Convertimos el texto del producto en su vector numérico
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

        # Guardamos todo el lote en ChromaDB en una sola operación
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Indexación completada: {len(ids)} insumos persistidos en ChromaDB.")
        return len(ids)

    def buscar_producto(self, consulta: str, n_resultados: int = 1) -> List[Dict[str, Any]]:
        """
        PASO 3: BÚSQUEDA SEMÁNTICA INTELIGENTE
        
        ¿Qué hace?
        1. Toma la consulta escrita por el cocinero (ej: 'posta limpia').
        2. La convierte en un vector matemático usando Gemini.
        3. Le pide a ChromaDB que busque los 'n' productos cuyos vectores estén
           más cerca en significado geométrico.
        4. Calcula el porcentaje de similitud (0% a 100%).
        
        ¿Qué devuelve?
        - Una lista con las mejores coincidencias del catálogo, incluyendo código SKU,
          nombre oficial, proveedor, precio y porcentaje de similitud.
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
                # En ChromaDB con distancia coseno: distancia = 1 - similitud
                # Una distancia cercana a 0 significa máxima afinidad
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


# ==============================================================================
# EJECUCIÓN DIRECTA (DEMOSTRACIÓN DE BÚSQUEDA SEMÁNTICA)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" INICIANDO INDEXADOR Y BÚSQUEDA VECTORIAL DEL CATÁLOGO HORECA")
    print("=" * 70)

    indexer = CatalogIndexer()
    total_indexados = indexer.indexar_catalogo(ruta_json="data/catalogo.json")

    print("\n" + "=" * 70)
    print(" PRUEBAS DE BÚSQUEDA SEMÁNTICA CON TÉRMINOS INFORMALES DE COCINA:")
    print("=" * 70)

    # Frases típicas de cocineros que no coinciden exactamente con el nombre de catálogo
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
            print(f"\n[Consulta Informal]: '{termino}'")
            print(f"  -> Coincidencia Oficial: [{meta.get('sku')}] {meta.get('nombre')}")
            print(f"  -> Proveedor Asignado:   {meta.get('proveedor')} (${meta.get('precio_actual'):,.0f} / {meta.get('unidad_minima')})")
            print(f"  -> Similitud Semántica:  {top['similitud'] * 100:.1f}%")
        else:
            print(f"\n[Consulta Informal]: '{termino}' -> Sin resultados.")