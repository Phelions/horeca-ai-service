"""
test_enrutamiento.py - Pruebas unitarias para enrutamiento condicional (Bloque 19 / U2.3).

Verifica exhaustivamente cada rama de RunnableBranch:
- Ruta de queja
- Ruta de pregunta
- Ruta de sugerencia
- Ruta por defecto (clasificaciones imprevistas)
- Normalización del clasificador (lower + strip)
- Preservación de la entrada original con RunnablePassthrough.assign
"""

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from enrutamiento import (
    construir_cadena_con_enrutamiento,
    construir_paso_clasificacion,
)


def test_enrutamiento_va_a_cadena_queja():
    """Valida que un mensaje clasificado como 'queja' se desvíe a cadena_queja."""
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.return_value = "queja"
    cadena_queja_falsa = MagicMock()
    cadena_queja_falsa.return_value = "respuesta de queja"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
        cadena_queja=cadena_queja_falsa,
    )
    resultado = cadena.invoke({"mensaje": "esto no funciona nunca"})

    assert cadena_queja_falsa.called
    assert resultado == "respuesta de queja"


def test_enrutamiento_va_a_cadena_pregunta():
    """Valida que un mensaje clasificado como 'pregunta' se desvíe a cadena_pregunta."""
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.return_value = "pregunta"
    cadena_pregunta_falsa = MagicMock()
    cadena_pregunta_falsa.return_value = "respuesta de pregunta"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
        cadena_pregunta=cadena_pregunta_falsa,
    )
    resultado = cadena.invoke({"mensaje": "¿cómo configuro las alertas de precios?"})

    assert cadena_pregunta_falsa.called
    assert resultado == "respuesta de pregunta"


def test_enrutamiento_va_a_cadena_sugerencia():
    """Valida que un mensaje clasificado como 'sugerencia' se desvíe a cadena_sugerencia."""
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.return_value = "sugerencia"
    cadena_sugerencia_falsa = MagicMock()
    cadena_sugerencia_falsa.return_value = "respuesta de sugerencia"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
        cadena_sugerencia=cadena_sugerencia_falsa,
    )
    resultado = cadena.invoke({"mensaje": "deberían permitir exportar los pedidos a Excel"})

    assert cadena_sugerencia_falsa.called
    assert resultado == "respuesta de sugerencia"


def test_enrutamiento_ruta_por_defecto():
    """
    Valida que ante una clasificación ambigua o imprevista, la ejecución caiga
    en la ruta por defecto (cadena_pregunta) sin lanzar excepciones.
    """
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.return_value = "categoría_desconocida"
    cadena_pregunta_falsa = MagicMock()
    cadena_pregunta_falsa.return_value = "respuesta por defecto"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
        cadena_pregunta=cadena_pregunta_falsa,
    )
    resultado = cadena.invoke({"mensaje": "texto sin categoría obvia"})

    assert cadena_pregunta_falsa.called
    assert resultado == "respuesta por defecto"


def test_paso_clasificacion_normaliza_texto():
    """
    Verifica que el clasificador aplique .strip().lower() sobre la salida del modelo
    para evitar fallos por mayúsculas o saltos de línea residuales.
    """
    modelo_mock = MagicMock()
    modelo_mock.return_value = AIMessage(content="  QUEJA \n")

    paso = construir_paso_clasificacion(modelo_mock)
    resultado = paso.invoke({"mensaje": "el pedido llegó incompleto"})

    assert resultado == "queja"


def test_passthrough_assign_conserva_mensaje_original():
    """
    Verifica que RunnablePassthrough.assign mantenga el diccionario de entrada
    original {'mensaje': ...} al agregar la clave 'categoria'.
    """
    clasificacion_mock = MagicMock()
    clasificacion_mock.return_value = "queja"

    ruta_captura = MagicMock()
    ruta_captura.side_effect = lambda x: f"mensaje={x['mensaje']}|categoria={x['categoria']}"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_mock,
        cadena_queja=ruta_captura,
    )
    resultado = cadena.invoke({"mensaje": "no llegó el despacho"})

    assert "mensaje=no llegó el despacho" in resultado
    assert "categoria=queja" in resultado
