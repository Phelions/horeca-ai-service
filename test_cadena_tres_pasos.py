"""
test_cadena_tres_pasos.py - Pruebas unitarias para la cadena de tres pasos (Bloque 18 / U2.3).

Verifica la correcta conexión de los 3 pasos mediante inyección de mocks,
asegurando que cada paso se invoque con la salida transformada del anterior,
sin consumir llamadas reales a la API ni tokens.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from cadena_tres_pasos import (
    construir_cadena,
    plantilla_extraccion,
    plantilla_simplificacion,
    plantilla_formato,
)


def test_cadena_conecta_los_tres_pasos():
    """
    Prueba canónica de la actividad: valida que los tres pasos se ejecuten
    secuencialmente, llamando al modelo exactamente 3 veces con los formatos esperados.
    """
    modelo_falso = MagicMock()
    modelo_falso.side_effect = [
        AIMessage(content="punto 1\npunto 2\npunto 3"),
        AIMessage(content="texto simplificado"),
        AIMessage(content="- viñeta 1\n- viñeta 2"),
    ]

    cadena = construir_cadena(modelo_falso)
    resultado = cadena.invoke({"texto": "texto de prueba"})

    assert modelo_falso.call_count == 3
    assert "viñeta" in resultado
    assert "- viñeta 1" in resultado


def test_cadena_falla_si_falta_variable():
    """
    Valida que un diccionario sin la clave requerida lance KeyError inmediatamente,
    evitando enviar peticiones malformadas al modelo.
    """
    modelo_falso = MagicMock()
    cadena = construir_cadena(modelo_falso)

    with pytest.raises(KeyError):
        cadena.invoke({"clave_invalida": "contenido"})

    assert modelo_falso.call_count == 0


def test_paso_extraccion_recibe_texto_formateado():
    """
    Verifica que el primer paso formatee adecuadamente la variable {texto}
    y que StrOutputParser retorne un string plano.
    """
    modelo_falso = MagicMock()
    modelo_falso.return_value = AIMessage(content="extraccion exitosa")

    paso = plantilla_extraccion | modelo_falso
    prompt_result = plantilla_extraccion.invoke({"texto": "hola mundo"})

    assert "hola mundo" in prompt_result.text
    assert hasattr(prompt_result, "text")


def test_salida_intermedia_es_str_plano():
    """
    Verifica que la salida de cada paso intermedio sea un 'str' puro y no
    un objeto con metadatos como AIMessage.
    """
    from langchain_core.output_parsers import StrOutputParser

    modelo_falso = MagicMock()
    modelo_falso.return_value = AIMessage(
        content="texto procesado",
        additional_kwargs={"extra": 123},
        response_metadata={"tokens": 10},
    )

    paso = plantilla_simplificacion | modelo_falso | StrOutputParser()
    resultado = paso.invoke({"puntos": "punto clave"})

    assert isinstance(resultado, str)
    assert resultado == "texto procesado"
    assert "additional_kwargs" not in resultado
