# Registro del Ensayo Funcional — Gate 2 (Unidad 2.8 / Bloque S1)

- **Fecha:** 2026-11-24
- **Proyecto:** `horeca-ai-service` (plataforma B2B que conecta negocios HORECA con proveedores mayoristas)
- **Alcance probado:** Integración modular de indexación persistida, recuperación semántica, filtro anti-inyección, citas de fuente, trazabilidad por `trace_id`, métricas de consumo/latencia y compuerta de supervisión humana LangGraph.

## Hallazgos detectados y corregidos durante la integración

1. **`trace_id: null` en el callback inicial del retriever (`logging_rag.py`)**
   - **Encontrado:** Al conectar `procesar_consulta_rag()` con el vector store y Gemini real, el primer evento `retriever_start` emitía `"trace_id": null` porque `trace_id_var` recién se asignaba dentro de `procesar_consulta()`, después de invocar al `retriever`.
   - **Corregido:** `procesar_consulta_rag()` ahora asigna `trace_id_var` desde antes de `retriever.invoke()` hasta el `finally` final, y `procesar_consulta()` reutiliza el `trace_id` activo cuando ya existe en el contexto.
   - **Resultado tras la corrección:** Los 7 eventos JSON de la consulta real (`retriever_start`, `retriever_end`, `consulta_recibida`, `llm_start`, `llm_end`, `consumo_llm`, `consulta_completada`) compartieron el mismo `trace_id` (`15d185a2ffc3`).

2. **Ausencia de corte antes del LLM cuando todo el contexto recuperado es descartado (`logging_rag.py`)**
   - **Encontrado:** Si `filtrar_documentos_sospechosos()` descartaba el 100 % de los fragmentos recuperados por contener patrones de inyección, `procesar_consulta()` igual habría llamado a `generador_fn` con un bloque de contexto vacío; además, el regex exigía `"prompt del sistema"` y no capturaba la variante `"revela tu prompt"`.
   - **Corregido:** Se amplió `PATRONES_SOSPECHOSOS` para cubrir `"revela tu prompt"` y se añadió un cortocircuito explícito previo a la invocación del modelo que registra `sin_contexto_confiable` en logs y devuelve `RESPUESTA_SIN_CONTEXTO` sin inventar respuesta ni consumir tokens.

3. **Formato de contenido en lista devuelto por Gemini (`logging_rag.py` / `servicio_rag.py`)**
   - **Encontrado:** La respuesta en vivo de `ChatGoogleGenerativeAI` devuelve bloques estructurados (`[{'type': 'text', 'text': ...}]`) en `.content`, mientras que `FakeListLLM` en modo offline devuelve `str`.
   - **Corregido:** Se incorporó `extraer_texto_respuesta()` y `_asegurar_cita_en_respuesta()` para normalizar ambos formatos y garantizar que la salida incluya tanto las citas `[N]` como los identificadores reales de fuente recuperados.

4. **Separación entre índice real (Gemini embeddings) e índice determinista de pruebas (`indexar_corpus.py` / `servicio_rag.py`)**
   - **Encontrado:** Reutilizar `vectorstore/` (indexado con `models/gemini-embedding-001`) con `DeterministicFakeEmbedding(size=768)` mezclaría dimensiones incompatibles o duplicaría documentos si se llamara `from_documents` en cada arranque.
   - **Corregido:** Se implementó `cargar_o_crear_indice()` y `actualizar_indice_incremental()` en `indexar_corpus.py`, manteniendo `vectorstore/` para corridas reales (`--real`) y `vectorstore_gate2/` (ignorado en `.gitignore`) para corridas offline reproducibles.
