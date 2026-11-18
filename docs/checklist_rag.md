# 📋 Checklist del Flujo RAG — HORECA AI Service (Unidad 2.5)

> **Auditoría Completa del Flujo RAG (Bloques 24 a 28)**  
> Verificación de extremo a extremo sobre la plataforma B2B que conecta negocios del sector HORECA (hoteles, restaurantes y cafeterías) con proveedores mayoristas.

---

## 1. 📦 Indexación (Bloques 24 y 25)

- **Estrategia de chunking elegida con evidencia (`test_chunking.py` / `indexar_corpus.py`):**
  - Se comparó en vivo `CharacterTextSplitter` (corte fijo a 110 caracteres, que partió la palabra `"Los errores"` en `"Los err"` / `"ores"`) contra `RecursiveCharacterTextSplitter`, eligiendo este último con `chunk_size=200` y `chunk_overlap=30`.
  - Los **10 documentos** técnicos del dominio HORECA (`CORPUS_HORECA`, de ~128 a ~165 caracteres cada uno) generaron exactamente **10 chunks** (`doc0` a `doc9`), conservando cada idea completa sin cortes de palabras.
- **Integridad de chunks verificada antes de indexar (`verificar_integridad_chunks`):**
  - Chunks vacíos: `[]` (`0`).
  - Duplicados: `False`.
  - Total de caracteres en chunks coincidente con el corpus original.
- **Consistencia de `persist_directory` y colección:**
  - Todos los módulos (`indexar_corpus.py`, `recuperacion.py`, `evaluacion_recuperacion.py`, `registro_rag.py`) comparten `PERSIST_DIRECTORY = "vectorstore/"` y `COLLECTION_NAME = "corpus_horeca"`.
  - Al reindexar, `indexar_corpus.py` limpia la caché del cliente Chroma y el directorio previo para evitar duplicación de chunks.
  - Al reconectar en disco con `cargar_vectorstore()`, `_collection.count()` devolvió `10` y `get()["ids"]` devolvió `['doc0', ..., 'doc9']`.
- **Exclusión en `.gitignore` y verificación `preflight.py`:**
  - `vectorstore/` y `vectorstore_ejemplo/` figuran en `.gitignore`.
  - `.venv/bin/python preflight.py` imprime: `Preflight OK: la carpeta del indice existe y esta en .gitignore`.

---

## 2. 🔎 Recuperación Semántica (Bloque 26)

- **Metadata de fuente indexada desde el origen:**
  - Cada documento se indexó con `METADATAS_HORECA` (`{"fuente": "..."}`) referenciando los archivos reales del repositorio (`cliente.py`, `src/consumo_api.py`, `src/order_parser.py`, `enrutamiento.py`, `orquestacion.py`, `agente_con_herramientas.py`, `docs/limites_autonomia_agente.md`, `tests/test_order_parser.py`).
  - El diccionario `citas` mapea claves `1..k` sin valores `"desconocida"`.
- **Criterio de selección de `k`:**
  - Con `k=3` y un documento relevante por pregunta en el conjunto actual, `evaluacion_recuperacion.py` alcanzó `recall=1.000` (el documento objetivo apareció siempre dentro del top-3), manteniendo el prompt compacto frente a `k=6`.
- **Métrica de distancia de Chroma:**
  - Chroma ordena resultados por **distancia** (menor distancia = mayor similitud semántica).

---

## 3. 📝 Generación con Citas (Bloque 26)

- **Instrucción explícita de cita (`PROMPT_TEMPLATE` en `recuperacion.py`):**
  - El prompt exige responder usando **solo** el contexto numerado (`[1]`, `[2]`, `[3]`) y citar cada afirmación con su número entre corchetes.
- **Manejo explícito de contexto insuficiente:**
  - `PROMPT_TEMPLATE` instruye explícitamente: *"Si el contexto no alcanza para responder, dilo explicitamente en vez de inventar"*.
  - `extraer_citas_de_respuesta` valida los índices citados con `citas.get(idx, "Cita invalida / no encontrada")` para evitar errores si el modelo alucina un número fuera de rango.

---

## 4. 📊 Evaluación Cuantitativa (Bloque 27)

- **Conjunto de evaluación independiente (`casos_evaluacion` en `evaluacion_recuperacion.py`):**
  - **10 preguntas** etiquetadas de antemano (`doc0`..`doc9`), cubriendo todas las piezas de la plataforma HORECA (cliente HTTP, errores 401, costos por tokens, parser Pydantic, enrutamiento, fallbacks, herramientas de riesgo bajo/medio/alto y pruebas).
- **Resultados reales medidos (`resultados_evaluacion.txt`, fecha `2026-11-18`, `k=3`, `chunk_size=200`, `chunk_overlap=30`):**
  - **10/10** consultas lograron `precision=0.33` y `recall=1.00`.
  - Promedio general registrado: `precision: 0.333`, `recall: 1.000`.
  - Con `DeterministicFakeEmbedding` y `mocks` también se validaron casos borde (`precision_at_k([], ...)` -> `0.0`, `recall_at_k(..., set())` -> `None`) y el trade-off `k=3` vs `k=5`.

---

## 5. ⏱️ Costo y Latencia Real (Bloque 28)

Medición por etapa en vivo con `models/gemini-embedding-001` + `gemini-3.8-flash` (`registro_rag.py` -> `registros_latencia_costo.jsonl`, leídos con `leer_registros.py`):

| Pregunta (resumen) | `k` | `embed_y_busqueda` (ms) | `generacion` (ms) | `tiempo_total_ms` | `input_tokens` | `output_tokens` | `total_tokens` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `que hace el cliente si una peticion tarda demasiado...` | 3 | 1230.69 | 5048.73 | 6279.42 | 192 | 404 | 596 |
| `como se calcula el costo en tokens de cada llamada...` | 3 | 1221.78 | 4440.99 | 5662.76 | 187 | 364 | 551 |
| `que acciones de alto riesgo exigen aprobacion humana...` | 3 | 1207.66 | 5248.84 | 6456.50 | 183 | 418 | 601 |
| `que hace el cliente si una peticion tarda demasiado...` | 6 | 1283.22 | 4445.42 | 5728.64 | 304 | 354 | 658 |

**Conclusiones observadas de las corridas reales:**
1. **Impacto de `k` en el costo:** Al duplicar `k` de `3` a `6` para la misma pregunta, los `input_tokens` reales subieron de **192** a **304** (+58.3% de entrada).
2. **Cuello de botella de latencia:** En las 4 ejecuciones, `generacion` (`4440.99`–`5248.84 ms`) fue consistentemente la etapa más lenta (~3.5x–4.3x mayor que `embed_y_busqueda`, que promedió ~`1235 ms`), confirmando que optimizar el prompt/modelo impacta más la latencia percibida que la búsqueda local en Chroma.
