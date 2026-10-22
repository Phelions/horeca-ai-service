"""
================================================================================
MÓDULO: prompts/banco.py
PROPÓSITO: Banco centralizado y versionado de prompts para el sistema HORECA.
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Guarda en un solo lugar los 3 prompts fundamentales de la aplicación.
En lugar de tener textos largos regados por todo el código, los centralizamos aquí
para que cualquier desarrollador pueda revisarlos, ajustarlos y probarlos con facilidad.

¿POR QUÉ CADA PROMPT TIENE ESTA ESTRUCTURA?
-------------------------------------------
Cada elemento del banco contiene:
1. 'nombre': Identificador único de la tarea (ej: extraer_pedido_horeca).
2. 'descripcion': Breve explicación en español de para qué sirve.
3. 'system_instruction': La regla o rol que le damos a Gemini en un canal separado
   (le dice quién es y cómo debe comportarse, sin competir con la pregunta del usuario).
4. 'ejemplo_usuario': Un mensaje real típico de cocina usado como caso de prueba.
5. 'claves_esperadas': La lista de nombres de campos que obligatoriamente deben
   venir en el JSON de respuesta.
6. 'response_schema': El esquema estricto que le pasamos a la API de Gemini para
   que no devuelva texto libre, sino un objeto estructurado.
"""

BANCO_DE_PROMPTS = [
    {
        "nombre": "extraer_pedido_horeca",
        "descripcion": "Extrae de forma limpia insumos, cantidades numéricas y unidades desde mensajes informales de cocina.",
        "system_instruction": (
            "Eres un asistente experto en adquisiciones para el sector gastronómico (HORECA). "
            "Tu misión es analizar mensajes informales de cocina y extraer cada insumo solicitado, "
            "su cantidad numérica precisa (usando decimales si es fraccionario, ej: 0.5) y su unidad de medida. "
            "Responde estrictamente en formato JSON válido según el esquema solicitado, sin bloques de código markdown "
            "ni texto conversacional adicional."
        ),
        "ejemplo_usuario": (
            "Hola don Carlos, para mañana necesitamos con urgencia 2 cajas de tomate limachino bien firme, "
            "3 sacos de harina panadera, unos 10 kilos de posta negra limpia y 2 bidones de lavaloza concentrado. "
            "Ah y si tiene palta hass, mándeme 1 caja. Gracias!"
        ),
        "claves_esperadas": ["items"],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "producto_mencionado": {"type": "STRING"},
                            "cantidad": {"type": "NUMBER"},
                            "unidad_mencionada": {"type": "STRING"},
                            "observaciones": {"type": "STRING"}
                        },
                        "required": ["producto_mencionado", "cantidad", "unidad_mencionada"]
                    }
                }
            },
            "required": ["items"]
        }
    },
    {
        "nombre": "clasificar_urgencia_pedido",
        "descripcion": "Evalúa la urgencia horaria y la categoría de insumos para priorizar la logística de compra matutina.",
        "system_instruction": (
            "Eres un despachador logístico para proveedores mayoristas de restaurantes. "
            "Analiza el mensaje del cliente y determina el nivel de urgencia ('ALTA', 'MEDIA', 'BAJA'), "
            "el horario límite requerido de entrega, la categoría principal de insumos y una justificación breve. "
            "Responde exclusivamente en formato JSON estructurado."
        ),
        "ejemplo_usuario": (
            "Carlos, necesitamos urgente antes de las 8:30 AM 5 kilos de pechuga deshuesada y 3 bandejas de champiñones "
            "porque nos quedamos sin stock para el servicio del almuerzo."
        ),
        "claves_esperadas": ["nivel_urgencia", "horario_limite", "categoria_insumo", "justificacion"],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "nivel_urgencia": {"type": "STRING"},
                "horario_limite": {"type": "STRING"},
                "categoria_insumo": {"type": "STRING"},
                "justificacion": {"type": "STRING"}
            },
            "required": ["nivel_urgencia", "horario_limite", "categoria_insumo", "justificacion"]
        }
    },
    {
        "nombre": "resumen_orden_compra",
        "descripcion": "Genera una síntesis ejecutiva para el administrador con conteo total de ítems y alertas de conservación.",
        "system_instruction": (
            "Eres un auditor de compras gastronómicas. Analiza el pedido solicitado y genera un resumen ejecutivo "
            "indicando la cantidad total de insumos distintos pedidos, si el despacho incluye productos perecibles "
            "que requieran cadena de frío (booleano), y si hay alguna alerta de calidad o maduración señalada. "
            "Responde únicamente con un objeto JSON válido según el esquema."
        ),
        "ejemplo_usuario": (
            "Para mañana: 2 cajas de tomate limachino bien maduro, 1 saco de papas seleccionadas, "
            "10 kilos de posta negra limpia y 2 bidones de cloro desinfectante."
        ),
        "claves_esperadas": ["total_insumos", "resumen_ejecutivo", "requiere_frio", "alertas_calidad"],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "total_insumos": {"type": "NUMBER"},
                "resumen_ejecutivo": {"type": "STRING"},
                "requiere_frio": {"type": "BOOLEAN"},
                "alertas_calidad": {"type": "STRING"}
            },
            "required": ["total_insumos", "resumen_ejecutivo", "requiere_frio", "alertas_calidad"]
        }
    }
]
