"""
enrutamiento.py - Enrutamiento condicional según el tipo de entrada (Bloque 19 / U2.3).

Implementa la bifurcación condicional de flujos con LangChain usando
RunnableBranch y RunnablePassthrough.assign:
1. Clasifica el mensaje de entrada (pregunta, queja o sugerencia).
2. Normaliza el texto de la categoría (strip + lower) para comparaciones robustas.
3. Preserva el mensaje original junto a la categoría mediante RunnablePassthrough.assign.
4. Enruta dinámicamente hacia la sub-cadena especializada correspondiente,
   con una ruta por defecto ante categorías no previstas o ambiguas.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableLambda,
)

# 1. Plantilla de clasificación
plantilla_clasificacion = PromptTemplate.from_template(
    "Clasifica el siguiente mensaje de soporte en una sola "
    "palabra: pregunta, queja o sugerencia. Responde solo "
    "con esa palabra.\n\n{mensaje}"
)

# 2. Plantillas de ramas especializadas
plantilla_pregunta = PromptTemplate.from_template(
    "Responde esta pregunta técnica con precisión:\n\n{mensaje}"
)

plantilla_queja = PromptTemplate.from_template(
    "Responde a esta queja con empatía, reconociendo el "
    "problema antes de ofrecer una solución:\n\n{mensaje}"
)

plantilla_sugerencia = PromptTemplate.from_template(
    "Agradece esta sugerencia y explica cómo se evaluará "
    "dentro del producto:\n\n{mensaje}"
)


def construir_paso_clasificacion(modelo):
    """
    Construye el paso de clasificación acoplado al modelo suministrado,
    garantizando la normalización a minúsculas y sin espacios residuales.
    """
    return (
        plantilla_clasificacion
        | modelo
        | StrOutputParser()
        | (lambda texto: texto.strip().lower().strip("'\""))
    )


def construir_cadena_con_enrutamiento(
    paso_clasificacion,
    cadena_queja=None,
    cadena_sugerencia=None,
    cadena_pregunta=None,
):
    """
    Construye el flujo bifurcado usando RunnableBranch y RunnablePassthrough.assign.

    Asigna un RunnablePassthrough() por defecto a las rutas no especificadas,
    permitiendo inyectar mocks puntuales en pruebas unitarias para aislar cada camino.
    """
    cadena_queja = cadena_queja or RunnablePassthrough()
    cadena_sugerencia = cadena_sugerencia or RunnablePassthrough()
    cadena_pregunta = cadena_pregunta or RunnablePassthrough()

    bifurcacion = RunnableBranch(
        (lambda x: x["categoria"] == "queja", cadena_queja),
        (lambda x: x["categoria"] == "sugerencia", cadena_sugerencia),
        cadena_pregunta,
    )

    return (
        RunnablePassthrough.assign(categoria=paso_clasificacion)
        | bifurcacion
    )


# 3. Respuesta genérica y manejo de fallbacks (Bloque 20)
respuesta_generica = RunnableLambda(
    lambda x: (
        "Gracias por tu mensaje. Un miembro de nuestro "
        "equipo te va a responder pronto."
    )
)


def construir_cadena_queja_segura(cadena_queja):
    """
    Protege la ruta de quejas asociando un fallback genérico ante caídas del modelo.
    """
    return cadena_queja.with_fallbacks([respuesta_generica])


def procesar_mensaje_seguro(cadena, mensaje: str) -> str:
    """
    Red de seguridad global (try/except):
    Captura cualquier excepción no atrapada por fallbacks locales y garantiza
    que el usuario siempre reciba una respuesta comprensible sin botar la aplicación.
    """
    try:
        return cadena.invoke({"mensaje": mensaje})
    except Exception as error:
        print(f"Error procesando mensaje: {error}")
        return (
            "No pudimos procesar tu mensaje en este "
            "momento. Intenta de nuevo más tarde."
        )


def construir_cadenas_especializadas(modelo):
    """Genera las cadenas completas asociadas a un modelo concreto."""
    cadena_preg = plantilla_pregunta | modelo | StrOutputParser()
    cadena_quej = plantilla_queja | modelo | StrOutputParser()
    cadena_sug = plantilla_sugerencia | modelo | StrOutputParser()
    return cadena_preg, cadena_quej, cadena_sug


if __name__ == "__main__":
    load_dotenv()
    mensaje_ejemplo = "La app se cierra cada vez que subo una foto de la factura mayorista."
    print("--- Probando enrutamiento condicional con mensaje de queja ---")

    try:
        modelo = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
        paso_clasif = construir_paso_clasificacion(modelo)
        c_pregunta, c_queja, c_sugerencia = construir_cadenas_especializadas(modelo)

        cadena_con_enrutamiento = construir_cadena_con_enrutamiento(
            paso_clasificacion=paso_clasif,
            cadena_queja=c_queja,
            cadena_sugerencia=c_sugerencia,
            cadena_pregunta=c_pregunta,
        )
        resultado = cadena_con_enrutamiento.invoke({"mensaje": mensaje_ejemplo})
        print("\nRespuesta obtenida:")
        print(resultado)
    except Exception as e:
        print(f"\n[Aviso]: No se pudo contactar a la API remota ({e}).")
        print("Demostrando enrutamiento local con mock de clasificación e inyección de rama:")
        from unittest.mock import MagicMock

        clasif_mock = MagicMock()
        clasif_mock.return_value = "queja"
        queja_mock = MagicMock()
        queja_mock.return_value = (
            "Lamento mucho el inconveniente con la subida de fotos de la factura. "
            "Estamos derivando el caso a soporte técnico prioritario para resolverlo."
        )

        cadena_demo = construir_cadena_con_enrutamiento(
            paso_clasificacion=clasif_mock,
            cadena_queja=queja_mock,
        )
        print(f"\nEntrada: '{mensaje_ejemplo}'")
        print("Clasificación detectada: 'queja' -> Derivado a: cadena_queja")
        print("Respuesta obtenida:")
        print(cadena_demo.invoke({"mensaje": mensaje_ejemplo}))

