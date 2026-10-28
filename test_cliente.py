"""
Modulo 2 - Bloque 14: test_cliente.py
Suite de pruebas unitarias para el cliente modular con pytest y unittest.mock.
Verifica las 4 funciones críticas (enviar_peticion, con_reintento, registrar_costo, consultar_modelo)
sin consumir cuotas reales ni depender de internet.
"""

import pytest
from unittest.mock import patch, Mock

import cliente
from cliente import (
    enviar_peticion,
    con_reintento,
    registrar_costo,
    consultar_modelo,
)


@pytest.fixture(autouse=True)
def reiniciar_costo_total():
    """
    Fixture que se ejecuta automáticamente antes de cada prueba
    para evitar que el acumulado de una prueba afecte a la siguiente.
    """
    cliente.costo_total_sesion = 0.0
    yield


# ==============================================================================
# 1. Pruebas de enviar_peticion (Clasificación de errores y respuesta HTTP)
# ==============================================================================

def test_enviar_peticion_credencial_invalida():
    """Verifica que un código HTTP 401 lance ValueError (no reintentable)."""
    respuesta_falsa = Mock(status_code=401)

    with patch("cliente.requests.post", return_value=respuesta_falsa):
        with pytest.raises(ValueError, match="Credencial inválida"):
            enviar_peticion({"pregunta": "test"})


def test_enviar_peticion_limite_de_uso():
    """Verifica que un código HTTP 429 lance TimeoutError (reintentable)."""
    respuesta_falsa = Mock(status_code=429)

    with patch("cliente.requests.post", return_value=respuesta_falsa):
        with pytest.raises(TimeoutError, match="El proveedor no pudo procesar"):
            enviar_peticion({"pregunta": "test"})


def test_enviar_peticion_exito():
    """Verifica que un código HTTP 200 devuelva el contenido JSON tal cual."""
    respuesta_falsa = Mock(status_code=200)
    respuesta_falsa.json.return_value = {"texto": "respuesta simulada"}

    with patch("cliente.requests.post", return_value=respuesta_falsa):
        resultado = enviar_peticion({"pregunta": "test"})

    assert resultado == {"texto": "respuesta simulada"}


# ==============================================================================
# 2. Pruebas de con_reintento (Lógica de reintentos y excepciones)
# ==============================================================================

def test_con_reintento_agota_intentos():
    """Verifica que al agotarse los 3 intentos, se propague el TimeoutError original."""
    def siempre_falla():
        raise TimeoutError("Límite de uso simulado")

    with patch("cliente.time.sleep") as mock_sleep:
        with pytest.raises(TimeoutError):
            con_reintento(siempre_falla, intentos=3)

    # Debe haber llamado a sleep 2 veces (entre intentos 1-2 y 2-3)
    assert mock_sleep.call_count == 2


def test_con_reintento_no_reintenta_value_error():
    """Verifica que si ocurre un ValueError (credencial inválida), NO se reintente."""
    intentos = {"conteo": 0}

    def falla_credencial():
        intentos["conteo"] += 1
        raise ValueError("Clave inválida")

    with patch("cliente.time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="Clave inválida"):
            con_reintento(falla_credencial, intentos=3)

    # Solo debe ejecutarse 1 vez y no llamar a sleep
    assert intentos["conteo"] == 1
    mock_sleep.assert_not_called()


# ==============================================================================
# 3. Pruebas de registrar_costo (Auditoría de tokens y acumulación de sesión)
# ==============================================================================

def test_registrar_costo_acumula():
    """Verifica el cálculo de costo por tokens y su suma a costo_total_sesion."""
    respuesta_falsa = {"usageMetadata": {"totalTokenCount": 1000}}
    costo = registrar_costo(respuesta_falsa)

    assert costo == cliente.COSTO_POR_1000_TOKENS
    assert cliente.costo_total_sesion == cliente.COSTO_POR_1000_TOKENS


# ==============================================================================
# 4. Pruebas de consultar_modelo (Traducción de errores para el consumidor)
# ==============================================================================

def test_consultar_modelo_convierte_error():
    """Verifica que cuando los reintentos se agotan, consultar_modelo lance RuntimeError."""
    error_simulado = TimeoutError("simulado")

    with patch("cliente.con_reintento", side_effect=error_simulado):
        with pytest.raises(RuntimeError, match="El servicio no respondió tras varios intentos"):
            consultar_modelo("pregunta de prueba")
