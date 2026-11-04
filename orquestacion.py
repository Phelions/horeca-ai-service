"""
orquestacion.py - Flujo orquestado end-to-end con LangChain (Bloque 20 / U2.3).

Integra los tres pilares de la orquestación en LCEL aplicados a HORECA AI Service:
1. Encadenamiento secuencial multi-paso (extracción -> normalización -> formato).
2. Enrutamiento condicional con RunnableBranch y preservación de estado con RunnablePassthrough.assign
   (pedidos gastronómicos, reclamos de proveedores y consultas generales).
3. Resiliencia y manejo de errores por paso mediante fallbacks (.with_fallbacks),
   asegurando que la indisponibilidad de un modelo o timeout no tumbe el flujo completo.
"""

from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableLambda,
)


# ============================================================================
# 1. PLANTILLAS DE CLASIFICACIÓN Y ENRUTAMIENTO (BLOQUE 19 & 20)
# ============================================================================

plantilla_clasificacion_horeca = PromptTemplate.from_template(
    "Clasifica el siguiente mensaje operativo de restaurante en una sola "
    "palabra: 'pedido', 'reclamo' o 'consulta'. Responde únicamente con esa palabra.\n\n{mensaje}"
)

plantilla_reclamo_horeca = PromptTemplate.from_template(
    "Eres un asistente de compras para restaurantes. Responde a este reclamo "
    "o discrepancia de proveedor con empatía, registrando el incidente y proponiendo "
    "una acción de contingencia:\n\n{mensaje}"
)

plantilla_consulta_horeca = PromptTemplate.from_template(
    "Eres un asistente de compras para restaurantes. Responde esta consulta "
    "de insumos o precios mayoristas de forma concisa y directa:\n\n{mensaje}"
)


# ============================================================================
# 2. PLANTILLAS PARA LA CADENA DE PEDIDOS (MULTI-PASO - BLOQUE 18 & 20)
# ============================================================================

plantilla_extraer_items = PromptTemplate.from_template(
    "Extrae únicamente los insumos, cantidades y unidades solicitados en el "
    "siguiente mensaje de compra gastronómica, uno por línea:\n\n{mensaje}"
)

plantilla_normalizar_items = PromptTemplate.from_template(
    "Normaliza los siguientes insumos a formato estándar de inventario "
    "(Producto, Cantidad Numérica, Unidad de Medida en kg/lt/unid):\n\n{items_crudos}"
)

plantilla_formatear_orden = PromptTemplate.from_template(
    "Convierte estos insumos normalizados en una orden de compra estructurada "
    "en viñetas markdown lista para enviar al proveedor:\n\n{items_normalizados}"
)


# ============================================================================
# 3. CONSTRUCTORES DE PASOS Y SUB-CADENAS RESILIENTES
# ============================================================================

def construir_paso_clasificacion(modelo):
    """
    Construye el clasificador normalizado (strip + lower) sobre la salida del modelo.
    """
    return (
        plantilla_clasificacion_horeca
        | modelo
        | StrOutputParser()
        | (lambda texto: texto.strip().lower().replace("'", "").replace('"', ""))
    )


def construir_cadena_pedido(modelo_primario, modelo_respaldo=None):
    """
    Construye la cadena de tres pasos para pedidos con tolerancia a fallos (.with_fallbacks).

    Si el modelo primario falla en la etapa de extracción o normalización (ej: timeout o error 429),
    el paso se recupera automáticamente ejecutando la ruta de respaldo sin tumbar todo el flujo.
    """
    paso1_primario = plantilla_extraer_items | modelo_primario | StrOutputParser()

    if modelo_respaldo is not None:
        paso1_respaldo = plantilla_extraer_items | modelo_respaldo | StrOutputParser()
        paso1 = paso1_primario.with_fallbacks([paso1_respaldo])
    else:
        # Fallback de contingencia heurística si no se suministra segundo modelo
        fallback_heuristico = RunnableLambda(
            lambda x: f"- Items extraídos (modo contingencia): {x.get('mensaje', '')}"
        )
        paso1 = paso1_primario.with_fallbacks([fallback_heuristico])

    paso2_primario = plantilla_normalizar_items | modelo_primario | StrOutputParser()
    if modelo_respaldo is not None:
        paso2_respaldo = plantilla_normalizar_items | modelo_respaldo | StrOutputParser()
        paso2 = paso2_primario.with_fallbacks([paso2_respaldo])
    else:
        paso2 = paso2_primario

    paso3 = plantilla_formatear_orden | (modelo_respaldo or modelo_primario) | StrOutputParser()

    cadena_pedido = (
        paso1
        | (lambda items_crudos: {"items_crudos": items_crudos})
        | paso2
        | (lambda items_normalizados: {"items_normalizados": items_normalizados})
        | paso3
    )
    return cadena_pedido


def construir_cadena_reclamo(modelo):
    """Construye la sub-cadena especializada en reclamos y no conformidades."""
    return plantilla_reclamo_horeca | modelo | StrOutputParser()


def construir_cadena_consulta(modelo):
    """Construye la sub-cadena especializada en consultas operativas rápidas."""
    return plantilla_consulta_horeca | modelo | StrOutputParser()


# ============================================================================
# 4. ORQUESTADOR PRINCIPAL (INTEGRACIÓN CORE)
# ============================================================================

def construir_flujo_orquestado(
    modelo_primario,
    modelo_respaldo=None,
    paso_clasificacion=None,
    cadena_pedido=None,
    cadena_reclamo=None,
    cadena_consulta=None,
):
    """
    Construye el flujo completo orquestado de extremo a extremo:
    - Validación y asignación de categoría sin perder el mensaje original.
    - Bifurcación condicional según la intención detectada.
    - Ejecución de cadenas multi-paso y recuperación transparente ante excepciones.
    """
    paso_clasif = paso_clasificacion or construir_paso_clasificacion(modelo_primario)

    c_pedido = cadena_pedido or construir_cadena_pedido(modelo_primario, modelo_respaldo)
    c_reclamo = cadena_reclamo or construir_cadena_reclamo(modelo_primario)
    c_consulta = cadena_consulta or construir_cadena_consulta(modelo_primario)

    bifurcacion = RunnableBranch(
        (lambda x: x["categoria"] == "pedido", c_pedido),
        (lambda x: x["categoria"] == "reclamo", c_reclamo),
        c_consulta,  # Ruta por defecto segura (para consultas o clasificaciones imprevistas)
    )

    flujo_completo = (
        RunnablePassthrough.assign(categoria=paso_clasif)
        | bifurcacion
    )
    return flujo_completo


if __name__ == "__main__":
    load_dotenv()
    print("=== DEMOSTRACIÓN END-TO-END DE FLUJO ORQUESTADO (U2.3) ===")
    mensaje_pedido = "Hola, envíame 5 cajas de tomate limachino y 3 sacos de cebolla blanca para mañana."
    mensaje_reclamo = "El pedido de ayer llegó con 2 cajas de palta podridas y el proveedor no contesta."

    try:
        modelo = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
        flujo = construir_flujo_orquestado(modelo_primario=modelo)

        print(f"\n[Entrada Pedido]: {mensaje_pedido}")
        salida_pedido = flujo.invoke({"mensaje": mensaje_pedido})
        print(f"[Salida Pedido]:\n{salida_pedido}")

        print(f"\n[Entrada Reclamo]: {mensaje_reclamo}")
        salida_reclamo = flujo.invoke({"mensaje": mensaje_reclamo})
        print(f"[Salida Reclamo]:\n{salida_reclamo}")
    except Exception as e:
        print(f"\n[Aviso]: No se pudo contactar a la API remota ({e}).")
        print("Demostrando orquestación local con modelos simulados y recuperación de fallos:")
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage

        # Simular clasificador
        clasif_demo = MagicMock()
        clasif_demo.side_effect = ["pedido", "reclamo"]

        # Simular modelo de pedidos
        modelo_pedido_demo = MagicMock()
        modelo_pedido_demo.side_effect = [
            AIMessage(content="tomate: 5 cajas\ncebolla: 3 sacos"),
            AIMessage(content="Tomate Limachino, 5, cajas\nCebolla Blanca, 3, sacos"),
            AIMessage(content="- 5 cajas Tomate Limachino\n- 3 sacos Cebolla Blanca"),
        ]
        cadena_ped_demo = construir_cadena_pedido(modelo_pedido_demo)

        cadena_recl_demo = MagicMock()
        cadena_recl_demo.return_value = (
            "Se registró el reclamo por '2 cajas de palta podridas'. "
            "Se notifica al proveedor y se solicita nota de crédito o reposición inmediata."
        )

        flujo_demo = construir_flujo_orquestado(
            modelo_primario=MagicMock(),
            paso_clasificacion=clasif_demo,
            cadena_pedido=cadena_ped_demo,
            cadena_reclamo=cadena_recl_demo,
        )

        print(f"\n1. Invocando con [Pedido]: '{mensaje_pedido}'")
        res_ped = flujo_demo.invoke({"mensaje": mensaje_pedido})
        print(f"Salida Orquestada:\n{res_ped}")

        print(f"\n2. Invocando con [Reclamo]: '{mensaje_reclamo}'")
        res_recl = flujo_demo.invoke({"mensaje": mensaje_reclamo})
        print(f"Salida Orquestada:\n{res_recl}")

