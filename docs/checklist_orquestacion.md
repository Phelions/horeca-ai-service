# 📋 Checklist de Orquestación con LangChain (U2.3 / Bloques 18, 19 y 20)

> **Documento de Auditoría y Verificación de Flujos Compuestos para HORECA AI Service**  
> *Especialización en Inteligencia Artificial Generativa — Módulo 2 / Unidad 2.3*

---

## 📌 1. Introducción y Propósito
El paso de clientes monolíticos basados en peticiones HTTP individuales a **arquitecturas orquestadas con LangChain Expression Language (LCEL)** introduce nuevos puntos de fallo potenciales si no se aplican patrones defensivos. Este documento sintetiza los criterios auditados y verificados en la implementación de `cadena_tres_pasos.py`, `enrutamiento.py` y `orquestacion.py`.

---

## 🔍 2. Puntos de Verificación del Checklist

| # | Criterio de Orquestación | Riesgo Asociado | Implementación y Solución en HORECA AI Service | Estado |
|---|---|---|---|---|
| **1** | **Alineación de Variables de Plantillas (`PromptTemplate`)** | `KeyError` en tiempo de ejecución si una lambda emite una clave distinta a la esperada por el template. | Verificación estricta de variables en cada paso mediante `re.findall(r"\{([a-zA-Z0-9_]+)\}", template)`. Si falta una clave obligatoria (ej: `'mensaje'`, `'puntos'`, `'texto_simple'`), falla tempranamente antes de invocar el modelo. | ✅ **Verificado** |
| **2** | **Aislamiento de Respuestas con `StrOutputParser`** | Fuga del objeto con metadatos (`AIMessage(content='...', additional_kwargs={})`) inyectado como string plano dentro del prompt siguiente. | Incorporación mandatoria de `StrOutputParser()` en todas las etapas intermedias de la cadena antes de suministrar el resultado al diccionario del paso posterior. | ✅ **Verificado** |
| **3** | **Normalización Defensiva en Clasificadores** | Fallo silencioso en comparaciones de cadenas (`categoria == "queja"`) por mayúsculas, comillas o saltos de línea devueltos por el LLM. | Aplicación de `.strip().lower().replace("'", "").replace('"', "")` al final del paso de clasificación, garantizando comparaciones deterministas en `RunnableBranch`. | ✅ **Verificado** |
| **4** | **Preservación de Estado con `RunnablePassthrough.assign`** | Pérdida del input original (`mensaje`) al ejecutar el paso de clasificación, dejando a las ramas especializadas sin datos. | Uso de `RunnablePassthrough.assign(categoria=paso_clasificacion)`, enriqueciendo el diccionario de contexto sin descartar el texto original del usuario. | ✅ **Verificado** |
| **5** | **Ruta por Defecto Segura en `RunnableBranch`** | Invocación que no empareja ninguna condición, lanzando error o retornando `None`. | Definición explícita de `cadena_consulta` / ruta por defecto al final de `RunnableBranch`, capturando entradas ambiguas o categorías no contempladas. | ✅ **Verificado** |
| **6** | **Tolerancia a Fallos por Paso (`with_fallbacks`)** | Que una caída temporal de red (Timeout) o cuota (429) en un paso intermedio tumbe toda la transacción del restaurante. | Enlace de modelos secundarios o respaldos heurísticos mediante `.with_fallbacks([paso_respaldo])` en las etapas críticas de extracción y normalización. | ✅ **Verificado** |
| **7** | **Inyección de Dependencias y Aislamiento de Pruebas** | Tests unitarios que llaman a la API real, consumen tokens y son no-deterministas. | Todas las funciones constructoras (`construir_cadena`, `construir_cadena_con_enrutamiento`, `construir_flujo_orquestado`) reciben el modelo o pasos como parámetros, posibilitando pruebas 100% deterministas con `MagicMock` y `AIMessage`. | ✅ **Verificado** |

---

## 🛠️ 3. Evidencia de Ejecución de Pruebas Automatizadas

La suite completa de pruebas unitarias (`test_cadena_tres_pasos.py`, `test_enrutamiento.py` y `test_orquestacion.py`) fue ejecutada satisfactoriamente en el entorno de desarrollo:

```bash
pytest test_cadena_tres_pasos.py test_enrutamiento.py test_orquestacion.py -v
```

### Resultados Obtenidos:
- `test_cadena_tres_pasos.py`:
  - `test_cadena_conecta_los_tres_pasos`: PASSED (3 llamadas exactas al modelo, extracción de viñetas).
  - `test_cadena_falla_si_falta_variable`: PASSED (`KeyError` temprano ante variable ausente).
  - `test_paso_extraccion_recibe_texto_formateado`: PASSED.
  - `test_salida_intermedia_es_str_plano`: PASSED (garantiza ausencia de metadatos de respuesta).
- `test_enrutamiento.py`:
  - `test_enrutamiento_va_a_cadena_queja`: PASSED.
  - `test_enrutamiento_va_a_cadena_pregunta`: PASSED.
  - `test_enrutamiento_va_a_cadena_sugerencia`: PASSED.
  - `test_enrutamiento_ruta_por_defecto`: PASSED.
  - `test_paso_clasificacion_normaliza_texto`: PASSED.
  - `test_passthrough_assign_conserva_mensaje_original`: PASSED.
- `test_orquestacion.py`:
  - `test_flujo_orquestado_ruta_pedido_ejecuta_tres_pasos`: PASSED.
  - `test_flujo_orquestado_ruta_reclamo`: PASSED.
  - `test_flujo_orquestado_ruta_por_defecto_ante_categoria_desconocida`: PASSED.
  - `test_resiliencia_with_fallbacks_en_paso_de_extraccion`: PASSED (recuperación automática ante Timeout/429).
  - `test_fallback_heuristico_cuando_no_hay_segundo_modelo`: PASSED.
  - `test_orquestador_requiere_clave_mensaje`: PASSED.
  - `test_normalizacion_de_clasificacion`: PASSED.

---

## 🚀 4. Conclusión
La arquitectura orquestada cumple con los más altos estándares de robustez exigidos para la operación 24/7 en cocinas y cadenas gastronómicas: separa claramente las responsabilidades, tipa las transformaciones intermedias y contiene fallos a nivel de paso sin degradar la disponibilidad global del sistema.
