"""
Banco de Prompts del Sistema HORECA AI Service
Contiene las instrucciones de sistema, esquemas de salida JSON estructurada
y ejemplos de entrada de usuario para las 3 tareas fundamentales de la miniaplicación.
"""

BANCO_DE_PROMPTS = [
    {
        "nombre": "extraer_pedido_horeca",
        "descripcion": "Extrae de forma quirúrgica insumos, cantidades numéricas y unidades de medida desde mensajes informales de cocina.",
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
        "descripcion": "Determina la criticidad temporal y la categoría de abastecimiento para priorizar despachos matutinos.",
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
        "descripcion": "Genera una síntesis ejecutiva para el administrador del restaurante con conteo de insumos y alertas de conservación.",
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
