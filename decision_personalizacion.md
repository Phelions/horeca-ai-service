# Decisión de personalización: plataforma B2B HORECA AI Service

## 1. El problema

**HORECA AI Service** conecta negocios del sector HORECA (hoteles, restaurantes y cafeterías) con proveedores mayoristas para digitalizar pedidos informales, cotizar insumos y resolver reclamos de despacho. Al procesar consultas reales, el sistema necesita reflejar información operativa que cambia continuamente (precios mayoristas, quiebres de stock, sustituciones válidas, formatos de envase como bidones de 10L/20L o sacos de 25kg/50kg, y certificaciones de alérgenos como ausencia de gluten), sin inventar datos cuando el catálogo no cubre la consulta y manteniendo respuestas JSON estructuradas con un costo acotado.

## 2. Resultado del método de decisión

Evaluación cuantitativa aplicada mediante `recomendar_estrategia(respuestas_horeca)` en `decision.py` usando evidencia verificable del repositorio:

| Criterio (`respuestas_horeca`) | Valor | Evidencia concreta en el proyecto | Impacto en puntaje |
| :--- | :---: | :--- | :--- |
| `informacion_cambia_seguido` | `True` | El catálogo de proveedores (`CATALOGO_INSUMOS_HORECA` en `agente_con_herramientas.py`), el corpus indexado (`indexar_corpus.py`) y la disponibilidad/precios cambian de forma semanal y mensual (quiebres y sustituciones vistos en `revision_manual.json`). | `+3 RAG` |
| `problema_de_formato_o_tono` | `False` | En `prompts/registro_resultados.json`, los 3 prompts del banco lograron `100.0 %` (`9/9` salidas válidas) y `src/consumo_api.py` (`parsear_respuesta_json`) ya resuelve texto introductorio. *(En `revision_manual.json`, `4 de 10` casos revisados fallaron por contenido/especificación o vocabulario, no por sintaxis JSON).* | `+0` |
| `dataset_grande_y_de_calidad` | `False` | Solo hay `12` ejemplos etiquetados en `dataset_tuning.py` (y `20` pares de evaluación en `resultados_metricas.json`), por debajo del umbral de `200` ejemplos. | `+0` |
| `presupuesto_limitado_o_iterar_rapido` | `True` | Se trabaja con `Free Gemini API Key` (sujeta a cuota `429 RESOURCE_EXHAUSTED` mitigada en `cliente.py` e `indexar_corpus.py`) y entregas semanales que requieren iterar en segundos/minutos (`preflight.py`, `indexar_corpus.py`). | `+2 RAG`, `+1 PEFT` |
| `necesita_multiples_variantes` | `False` | La plataforma opera un único servicio B2B unificado aislando conversaciones mediante `session_id` en `agente_con_herramientas.py`. | `+0` |

**Puntaje obtenido:**
- **RAG:** `5 puntos`
- **PEFT:** `1 punto`
- **Fine-tuning completo:** `0 puntos`

**Resultado:** No hubo empate (`candidatas = ['RAG']`).

**Análisis de sensibilidad (`analizar_sensibilidad`):** Al invertir una respuesta a la vez, `RAG` conserva el primer lugar en los 5 escenarios (`5` puntos frente a `0–3` de las alternativas, salvo si `informacion_cambia_seguido` pasara a `False`, donde baja a `2` puntos). La clave más sensible como posible duda es `problema_de_formato_o_tono`: si en una muestra futura de 20 solicitudes reales de tickets en texto libre al menos 3 incumplieran la clasificación esperada, pasar esa clave a `True` sube a **PEFT a 3 puntos** (`RAG: 5, PEFT: 3, Fine-tuning completo: 1`), posicionando a PEFT como complemento natural de segunda fase sobre la base RAG.

## 3. Técnica elegida y por qué

Se elige **RAG** como estrategia principal de personalización para **HORECA AI Service** (descartando **Fine-tuning completo**, **Soft-prompting** y posponiendo **PEFT** como complemento de Fase 2):

- **Por qué RAG resuelve el núcleo del problema:** El principal riesgo operativo detectado en `revision_manual.json` es de **conocimiento cambiante y específico por proveedor** (quiebres de stock, equivalencias de formato 10L vs 20L, harina fuerza 50kg vs estándar 25kg y alérgenos). Con RAG (ya instrumentado en `vectorstore/` con `recall: 1.000` en `resultados_evaluacion.txt`), actualizar una ficha o lista de precios toma segundos reindexando un chunk, sin reentrenar pesos ni memorizar hechos obsoletos. Además, el prompt numerado de `recuperacion.py` exige citar `[N]` y admitir cuando falta contexto.
- **Por qué se descartan las demás en esta etapa:**
  - *Fine-tuning completo (`0 puntos`):* Costoso, lento de iterar (horas por corrida) y requiere cientos de ejemplos; no actualiza conocimiento volátil de proveedores.
  - *Soft-prompting:* La API gestionada de Gemini (`google-genai==3.2.0`) no lo expone (`TuningMode` solo incluye `TUNING_MODE_FULL` y `TUNING_MODE_PEFT_ADAPTER`).
  - *PEFT (`1 punto`, eventual complemento):* Útil más adelante únicamente para fijar tono/formato en tareas estables como el clasificador de tickets de `dataset_tuning.py`, combinado con RAG una vez que exista volumen suficiente de ejemplos etiquetados.

## 4. Qué necesitarías para ejecutarla

1. **Ampliar el corpus productivo en `vectorstore/` (estrategia elegida: RAG):**
   - Hoy existen **10 documentos técnicos** indexados en `vectorstore/` (`indexar_corpus.py`, evaluados en `resultados_evaluacion.txt` con `precision: 0.333`, `recall: 1.000`).
   - Faltan **50 documentos reales** adicionales del negocio B2B para cubrir la operación mayorista antes del Gate 2 y producción (pasando de **10** a **60 documentos** en total): **20** fichas técnicas de insumos con formatos y alérgenos (harinas, aceites, lácteos, cárnicos y pescados), **15** listas de precios y reglas de despacho por proveedor, y **15** políticas de sustitución por quiebre de stock, cancelación y reclamos.
   - También se requiere ampliar el conjunto independiente de `evaluacion_recuperacion.py` de **10** a **25 preguntas** (sumando al menos **5** consultas sin respuesta en el corpus para verificar abstención explícita con `recall=None`) y ajustar `chunk_size`/`k` para mantener `recall >= 0.90` elevando `precision@k` desde `0.333`.
   - Infraestructura a mantener continuamente: respaldos de `vectorstore/` (Chroma) y proceso de reindexación automática ante altas/bajas de proveedores, asumiendo el costo por consulta ya medido en `registros_latencia_costo.jsonl` (~`183–192` tokens de entrada con `k=3`, ~`5.7–6.5 s` totales).

2. **Requisito cuantificado si se habilita PEFT como complemento en Fase 2 (`dataset_tuning.py`):**
   - Hoy el laboratorio `dataset_tuning.py` cuenta con **12 ejemplos** validados y perfectamente balanceados (**3** por cada una de las **4** categorías: `pedido_compra`, `reclamo_despacho`, `consulta_catalogo`, `ajuste_o_cancelacion`, `0` errores en `validar_dataset`).
   - Faltan **188 ejemplos** adicionales etiquetados y revisados para alcanzar el mínimo de **200 ejemplos** (**50** por categoría). Con `estimar_costo_de_etiquetar` (`5` minutos por ejemplo a `$25` USD/hora más una segunda revisión del `20 %` de los casos a `2` minutos), preparar esos **188** ejemplos faltantes tomaría **16.92 horas** de trabajo humano (**2.1 jornadas** de 8 horas, aprox. **$423.00 USD**).
