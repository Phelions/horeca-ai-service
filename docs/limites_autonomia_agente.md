# 🛡️ Límites de Autonomía y Auditoría del Agente HORECA AI Service

> **Documento Oficial de Gobierno, Seguridad y Trazabilidad Operativa**  
> *Especialización en Inteligencia Artificial Generativa — Unidad 2.4 (Bloques 21, 22 y 23 / Actividad Core)*

---

## Herramientas y nivel de riesgo

En una plataforma B2B que conecta negocios del sector HORECA (hoteles, restaurantes y cafeterías) con proveedores mayoristas, los fallos o alucinaciones de un agente autónomo conllevan impactos de diversa magnitud: desde una leve molestia informativa hasta mermas económicas severas, quiebres de inventario o compromisos contractuales involuntarios. Por ello, las herramientas expuestas al agente se clasifican estrictamente en **tres niveles de riesgo**:

| Nivel de Riesgo | Herramienta | Naturaleza de la Operación | Efectos Secundarios y Justificación Técnica | Esquema de Argumentos (`args_schema`) |
| :--- | :--- | :--- | :--- | :--- |
| **Bajo** | `consultar_catalogo_insumos` | Solo lectura (Consulta determinística) | Consulta de existencias, unidad de medida, proveedor y precio de referencia. No modifica base de datos ni afecta inventario físico. | `nombre_insumo: str` |
| **Bajo** | `calcular_cotizacion` | Cálculo determinístico | Proyección matemática de subtotales, IVA (19%) y total sin efectos colaterales ni reservas de presupuesto. | `EsquemaCalcularCotizacion` (`nombre_insumo`, `cantidad_unidades`) |
| **Medio** | `crear_borrador_pedido` | Escritura reversible | Registro preliminar de orden de compra en estado `BORRADOR`. No compromete presupuesto institucional ni emite contratos vinculantes con distribuidores mayoristas. Permite ajustes o descarte. | `EsquemaCrearBorradorPedido` (`proveedor`, `nombre_insumo`, `cantidad_unidades`) |
| **Medio** | `cancelar_borrador_pedido` | Escritura reversible | Transición de estado a `CANCELADO` de borradores no formalizados. Libera la intención de compra sin costo ni penalidad contractual. | `id_borrador: str` |
| **Alto** | `emitir_orden_compra_formal` | Escritura irreversible / Compromiso legal | Emisión vinculante de orden de compra comercial (`OC-2026-XXXX`). Compromete formalmente al negocio comprador con el proveedor y activa su logística de despacho. | `EsquemaEmitirOrdenCompra` (`id_borrador`, `proveedor`, `monto_total_pesos`) |
| **Alto** | `procesar_pago_proveedor` | Escritura irreversible / Impacto financiero directo | Transferencia monetaria bancaria irrevocable (`PAGO-2026-XXXX`) desde la cuenta del negocio comprador hacia el proveedor. Modificación directa de saldos. | `EsquemaProcesarPago` (`proveedor`, `monto_pesos`, `numero_factura`) |

### Criterios de Diseño Defensivo:
1. **Tipado Estricto con Pydantic:** Toda herramienta con más de un parámetro implementa un modelo de datos Pydantic `BaseModel` con `Field(description=...)` para guiar la inferencia del LLM y validar tipos antes de la ejecución.
2. **Aislamiento de Excepciones:** Todas las herramientas capturan internamente cualquier error con bloques `try/except Exception` devolviendo cadenas explicativas de tipo `str`, impidiendo que una falla en una herramienta aborte abruptamente el ciclo del agente.
3. **Registro de Traza Universal:** Cada herramienta vuelca atómicamente su acción y observación en el archivo de trazas auditable (`trazas_agente.jsonl`).

---

## Herramientas que requieren aprobación humana

El control de autonomía humana (*Human-In-The-Loop* o HITL) se aplica de forma selectiva para balancear la fluidez operativa con la seguridad corporativa:

### 1. Herramientas Sujetas a HITL Mandatorio
Exclusivamente las dos herramientas de **Riesgo Alto**:
* `emitir_orden_compra_formal`
* `procesar_pago_proveedor`

Las herramientas de riesgo bajo y medio se ejecutan de manera directa y autónoma sin solicitar confirmación humana interactiva.

### 2. Mecanismo de Control: `esperar_aprobacion`
La autorización explícita se implementa mediante la función:
```python
esperar_aprobacion(mensaje: str, tiempo_limite_segundos: int = 300) -> bool
```
* **Mecanismo I/O no bloqueante:** Emplea `select.select([sys.stdin], [], [], tiempo_limite_segundos)` para monitorear la entrada estándar con temporización precisa.
* **Política de Falla Segura (*Fail-Closed*):** Ante la expiración del tiempo límite (300 segundos por defecto), ingreso de opciones negativas (`n`, `no`), interrupción de teclado o cierre del flujo (`EOF`), la función retorna `False` inmediatamente.
* **Registro de Autorización:** Toda respuesta humana (afirmativa o denegada) se persiste en `registro_aprobaciones.jsonl` mediante `registrar_aprobacion`, garantizando la no-repudiación y el control de auditoría interna.

---

## Revisión de trazas

La observabilidad y el control continuo son indispensables para detectar desalineaciones, alucinaciones o degradaciones de comportamiento en modelos generativos.

### 1. Frecuencia y Muestreo de Auditoría
* **Operaciones de Riesgo Alto:** **100% de auditoría obligatoria**. Toda emisión de orden de compra formal y todo pago a proveedor debe auditarse cotejando la traza técnica con la autorización humana registrada.
* **Operaciones de Riesgo Bajo y Medio:** **Muestreo semanal sistemático** del 20% de las sesiones registradas en `trazas_agente.jsonl` (mínimo 50 trazas semanales estratificadas entre días hábiles y fines de semana).

### 2. Los Tres Patrones de Detección Críticos

1. **Patrón 1: Repeticiones Consecutivas de Herramientas (`herramientas[i] == herramientas[i - 1]`):**
   * *Descripción:* Ocurre cuando el agente entra en un bucle cerrado invocando la misma herramienta varias veces seguidas dentro de una misma sesión (ej. consulta reiterada del mismo insumo o intentos repetidos de generar un borrador).
   * *Causa común:* Incomprensión de la respuesta devuelta por la herramienta o formateo deficiente de la observación.
   * *Detección:* Función automatizada `marcar_repeticiones_sospechosas(ruta_archivo)` que inspecciona las secuencias y emite alertas detallando sesión y líneas de archivo.

2. **Patrón 2: Herramientas de Alto Riesgo sin Registro de Aprobación:**
   * *Descripción:* Detección en `trazas_agente.jsonl` de eventos asociados a `emitir_orden_compra_formal` o `procesar_pago_proveedor` con ejecución completada sin su contraparte autorizada (`aprobada: true`) en `registro_aprobaciones.jsonl`.
   * *Causa común:* Vulneración de flujo, fallo de lógica en herramientas personalizadas o bypass no autorizado.
   * *Detección:* Script de conciliación cruzada de folios transaccionales entre trazas y aprobaciones.

3. **Patrón 3: Respuestas Finales Contradictorias con las Observaciones:**
   * *Descripción:* El agente emite una respuesta al usuario afirmando el éxito de una operación comercial cuando la observación de la herramienta indicó un fallo (ej. la herramienta reportó *"Stock insuficiente"* o *"Operación cancelada por el operador humano"*, pero el agente afirma al comprador *"Su orden ha sido emitida exitosamente"*).
   * *Causa común:* Alucinación del modelo en la etapa final de generación tras ignorar el `ToolAgentAction` o la observación en `agent_scratchpad`.
   * *Detección:* Revisión semántica manual del muestreo semanal comparando `observacion` vs. `output`.

---

## Qué hacer ante un comportamiento inesperado

Si un supervisor, auditor o usuario detecta un comportamiento anómalo o sospechoso en el agente, se debe seguir estrictamente este protocolo secuencial:

```text
[Detección de Anomalía]
         │
         ▼
 1. Contención Inmediata  ────────► Suspender servicio del agente y revocar API keys
         │
         ▼
 2. Aislamiento de Trazas ────────► Ejecutar marcar_repeticiones_sospechosas y aislar sesión
         │
         ▼
 3. Verificación Financiera ──────► Cancelar borradores / Notificar banco y proveedores
         │
         ▼
 4. Diagnóstico y Corrección ─────► Corregir prompt, schemas Pydantic y umbrales HITL
         │
         ▼
 5. Pruebas y Reactivación ───────► Validar suite pytest al 100% antes de volver a producción
```

1. **Paso 1 — Contención Inmediata:**
   * Detener de inmediato el proceso del agente en producción para evitar invocaciones recursivas.
   * Si hay riesgo de transacciones bancarias, deshabilitar temporalmente las credenciales del conector financiero.

2. **Paso 2 — Aislamiento e Inspección Forense de Trazas:**
   * Extraer el archivo `trazas_agente.jsonl` y `registro_aprobaciones.jsonl`.
   * Ejecutar la herramienta `marcar_repeticiones_sospechosas` para identificar de forma precisa el identificador de sesión (`session_id`), la herramienta conflictiva y la secuencia temporal de acciones.

3. **Paso 3 — Verificación y Reversión Operativa (Rollback):**
   * *Para operaciones de Riesgo Medio:* Invocar `cancelar_borrador_pedido(id_borrador)` para anular cualquier borrador inconsistente generado por el agente.
   * *Para operaciones de Riesgo Alto:* Cotejar con compras y tesorería; de haberse formalizado una orden errónea, contactar de inmediato al proveedor gastronómico o a la entidad bancaria para cancelar la transferencia.

4. **Paso 4 — Diagnóstico y Ajuste de Salvaguardas:**
   * Analizar si la falla se debió a ambigüedad en el `PROMPT_SISTEMA_HORECA`, falta de restricciones en los `args_schema` de Pydantic, o falta de validación en la observación de la herramienta.
   * Modificar y robustecer las restricciones del prompt o esquemas de entrada.

5. **Paso 5 — Validación con Pruebas Automatizadas y Reactivación:**
   * Incorporar un nuevo caso de prueba en `test_agente_con_herramientas.py` que reproduzca el escenario fallido.
   * Ejecutar la suite completa (`pytest -v`) verificando una tasa de aprobación del 100%.
   * Restablecer el servicio en producción con supervisión reforzada durante las primeras 48 horas.

---

### 💻 Guía de Ejecución Fuera del Sandbox (Instalación, Ejecución y Git)

Para ejecutar, probar y auditar el agente en una estación de trabajo o servidor fuera del entorno sandbox restringido, siga estos comandos exactos:

#### 1. Preparación del Entorno Virtual e Instalación de Dependencias
```bash
# Navegar al directorio del repositorio
cd /home/system/horeca-ai-service

# Crear el entorno virtual si no existe
python3 -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate

# Instalar dependencias oficiales (incluyendo langchain-classic)
pip install -r requirements.txt
```

#### 2. Configuración de Credenciales
Asegúrese de configurar su clave de Gemini en el archivo `.env`:
```bash
cat << 'EOF' > .env
GEMINI_API_KEY=su_clave_api_aqui
GEMINI_MODEL=gemini-3.8-flash
LOG_LEVEL=DEBUG
EOF
```

#### 3. Ejecución Interactiva en Vivo (Generación de Trazas Reales)
Para interactuar con el agente en consola y generar eventos reales en `trazas_agente.jsonl` y `registro_aprobaciones.jsonl`:
```bash
# Ejecutar el agente en modo interactivo
python agente_con_herramientas.py
```
*Ejemplos de comandos para ingresar en la consola interactiva:*
* Saludo general (sin herramientas): `Hola, ¿quién eres y qué puedes hacer?`
* Consulta de bajo riesgo: `¿Cuánto stock tenemos de tomate triturado y a qué precio?`
* Cálculo de bajo riesgo: `Calcula la cotización de 20 kg de harina`
* Borrador de riesgo medio: `Crea un borrador para el proveedor Molino Central con 30 kg de harina`
* Cancelación de riesgo medio: `Cancela el borrador BORR-0001`
* Orden de compra de alto riesgo (requiere HITL): `Emite formalmente la orden de compra para el borrador BORR-0001 al proveedor Molino Central por 25500 pesos`

#### 4. Ejecución de la Suite de Pruebas Automatizadas
```bash
# Ejecutar todos los tests unitarios y de integración
pytest -v

# Ejecutar exclusivamente la suite del agente con herramientas
pytest test_agente_con_herramientas.py -v
```

#### 5. Comandos de Entrega Git
```bash
# Verificar estado de los archivos modificados y nuevos
git status

# Añadir los archivos de la implementación al área de preparación
git add agente_con_herramientas.py docs/limites_autonomia_agente.md test_agente_con_herramientas.py requirements.txt README.md

# Confirmar los cambios con un mensaje estructurado y descriptivo
git commit -m "feat(agente): implementa agente HORECA con herramientas en tres niveles de riesgo, memoria conversacional, HITL y auditoría de trazas (U2.4)"

# Publicar los cambios en la rama remota
git push origin main
```
