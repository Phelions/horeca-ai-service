"""
cadena_tres_pasos.py - Chain de tres pasos con LangChain (Bloque 18 / U2.3).

Implementa un flujo secuencial desacoplado con tres etapas diferenciadas:
1. Extraer los puntos clave de un texto (plantilla_extraccion).
2. Reescribir los puntos en lenguaje simple y directo (plantilla_simplificacion).
3. Formatear el resultado como una lista estructurada de viñetas markdown (plantilla_formato).

Diseñado para recibir el modelo como parámetro (inyección de dependencias),
permitiendo la ejecución con modelos reales (Gemini) o mocks en pruebas unitarias.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Definición de plantillas con responsabilidades atómicas y variables explícitas
plantilla_extraccion = PromptTemplate.from_template(
    "Lee el siguiente texto y devuelve solo sus 3 puntos "
    "clave, uno por línea, sin desarrollarlos:\n\n{texto}"
)

plantilla_simplificacion = PromptTemplate.from_template(
    "Reescribe estos puntos en lenguaje simple, sin jerga "
    "técnica, para alguien sin conocimientos previos:\n\n{puntos}"
)

plantilla_formato = PromptTemplate.from_template(
    "Convierte este texto en una lista de viñetas markdown, "
    "de máximo 3 líneas, sin introducción:\n\n{texto_simple}"
)


def construir_cadena(modelo):
    """
    Construye y retorna una chain LCEL de tres pasos encadenados con el operador '|'.

    Utiliza StrOutputParser en cada etapa intermedia para asegurar que el siguiente
    paso reciba texto plano ('str') en lugar de un objeto AIMessage con metadatos.
    Las funciones intermedias adaptan las claves del diccionario entre pasos.
    """
    paso_extraccion = plantilla_extraccion | modelo | StrOutputParser()
    paso_simplificacion = plantilla_simplificacion | modelo | StrOutputParser()
    paso_formato = plantilla_formato | modelo | StrOutputParser()

    cadena = (
        paso_extraccion
        | (lambda puntos: {"puntos": puntos})
        | paso_simplificacion
        | (lambda texto_simple: {"texto_simple": texto_simple})
        | paso_formato
    )
    return cadena


if __name__ == "__main__":
    load_dotenv()
    texto_abastecimiento = """
    En la gestión de abastecimiento gastronómico para restaurantes y cafeterías,
    los pedidos de insumos se realizan frecuentemente de noche vía notas de voz y mensajes informales.
    Esta práctica genera errores recurrentes de despacho, duplicación involuntaria de productos por ambigüedad
    y falta de conciliación oportuna entre los precios cotizados y los facturados por los mayoristas.
    Digitalizar este flujo mediante modelos de lenguaje permite extraer insumos de forma estructurada,
    estandarizar las unidades de compra y calcular costos estimados antes de emitir la orden formal al proveedor.
    """

    print("--- Ejecutando cadena de 3 pasos de extremo a extremo ---")
    try:
        modelo = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
        cadena = construir_cadena(modelo)
        resultado = cadena.invoke({"texto": texto_abastecimiento.strip()})
        print("\nResultado final:")
        print(resultado)
    except Exception as e:
        print(f"\n[Aviso]: No se pudo contactar a la API remota ({e}).")
        print("Demostrando ejecución local con inyección de respuestas estructuradas:")
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage

        modelo_demo = MagicMock()
        modelo_demo.side_effect = [
            AIMessage(content="Puntos extraídos:\n1. Pedidos informales causan errores.\n2. Falta de control en precios mayoristas.\n3. Digitalización permite orden y control de costos."),
            AIMessage(content="Pedir por mensajes informales lleva a errores y cobros incorrectos. Pasar a digital ayuda a pedir exacto y saber cuánto se gastará."),
            AIMessage(content="- Digitalización de compras gastronómicas para evitar mermas.\n- Estandarización de unidades de medida (kg, lt, unid).\n- Control previo de costos antes de emitir la orden."),
        ]
        cadena_demo = construir_cadena(modelo_demo)
        print("\nResultado final:")
        print(cadena_demo.invoke({"texto": texto_abastecimiento.strip()}))

