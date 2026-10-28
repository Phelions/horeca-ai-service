"""
test_cliente.py - Suite de pruebas unitarias para el cliente modular de HORECA AI Service.

Verifica con pytest y unittest.mock las 4 funciones críticas de resiliencia del Módulo 2:
1. enviar_peticion: Clasificación estricta de errores HTTP (401 -> ValueError, 429/5xx -> TimeoutError, 200 -> OK).
2. con_reintento: Reintentos con backoff ante TimeoutError y propagación directa ante ValueError.
3. registrar_costo: Auditoría y suma acumulativa de tokens en costo_total_sesion.
4. consultar_modelo: Orquestación de llamadas y traducción de TimeoutError en RuntimeError.

Ejecución rápida (< 0.2s), sin consumir cuotas de API ni esperar con pausas lentas.
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
    Fixture de pytest con autouse=True para reiniciar costo_total_sesion a 0.0
    antes de cada prueba y garantizar aislamiento total entre casos de prueba.
    """
    cliente.costo_total_sesion = 0.0
    yield


# ==============================================================================
# 1. PRUEBAS DE enviar_peticion (Clasificación de errores HTTP)
# ==============================================================================

def test_enviar_peticion_credencial_invalida():
    """Confirma que un error de autenticación (HTTP 401) lance ValueError (no reintentable)."""
    respuesta_falsa = Mock(status_code=401)
    payload_pedido = {"contents": [{"parts": [{"text": "2 cajas de tomate limachino"}]}]}

    with patch("cliente.requests.post", return_value=respuesta_falsa):
        with pytest.raises(ValueError, match="Credencial inválida o sin permiso"):
            enviar_peticion(payload_pedido)


def test_enviar_peticion_limite_de_uso():
    """Confirma que un exceso de cuota (HTTP 429) lance TimeoutError (sí reintentable)."""
    respuesta_falsa = Mock(status_code=429)
    payload_pedido = {"contents": [{"parts": [{"text": "10 kilos de posta negra"}]}]}

    with patch("cliente.requests.post", return_value=respuesta_falsa):
        with pytest.raises(TimeoutError, match="El proveedor no pudo procesar la petición"):
            enviar_peticion(payload_pedido)


def test_enviar_peticion_exito():
    """Confirma que una respuesta exitosa (HTTP 200) retorne el JSON procesado."""
    respuesta_falsa = Mock(status_code=200)
    respuesta_falsa.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "Pedido de 2 cajas de tomate confirmado."}]}}
        ]
    }

    with patch("cliente.requests.post", return_value=respuesta_falsa):
        resultado = enviar_peticion({"contents": [{"parts": [{"text": "2 cajas de tomate"}]}]})

    assert "candidates" in resultado
    assert resultado["candidates"][0]["content"]["parts"][0]["text"] == "Pedido de 2 cajas de tomate confirmado."


# ==============================================================================
# 2. PRUEBAS DE con_reintento (Lógica de recuperación y excepciones)
# ==============================================================================

def test_con_reintento_agota_intentos():
    """Confirma que si la llamada falla 3 veces seguidas por límite de cuota, propague TimeoutError."""
    def falla_cuota():
        raise TimeoutError("Límite de peticiones excedido en cocina")

    with patch("cliente.time.sleep") as mock_sleep:
        with pytest.raises(TimeoutError, match="Límite de peticiones excedido"):
            con_reintento(falla_cuota, intentos=3)

    # Con 3 intentos, debe pausar 2 veces entre reintentos
    assert mock_sleep.call_count == 2


def test_con_reintento_no_reintenta_value_error():
    """Confirma que si ocurre un ValueError (clave inválida), NO se reintente y termine de inmediato."""
    conteo_llamadas = {"ejecuciones": 0}

    def falla_clave():
        conteo_llamadas["ejecuciones"] += 1
        raise ValueError("Clave GEMINI_API_KEY incorrecta")

    with patch("cliente.time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="Clave GEMINI_API_KEY incorrecta"):
            con_reintento(falla_clave, intentos=3)

    # No debe reintentar: se detiene en la primera ejecución sin llamar a sleep
    assert conteo_llamadas["ejecuciones"] == 1
    mock_sleep.assert_not_called()


# ==============================================================================
# 3. PRUEBAS DE registrar_costo (Auditoría financiera de tokens)
# ==============================================================================

def test_registrar_costo_acumula():
    """Confirma que lea los tokens de usageMetadata y acumule el costo en costo_total_sesion."""
    respuesta_falsa = {"usageMetadata": {"totalTokenCount": 1000}}
    costo_operacion = registrar_costo(respuesta_falsa)

    assert costo_operacion == cliente.COSTO_POR_1000_TOKENS
    assert cliente.costo_total_sesion == cliente.COSTO_POR_1000_TOKENS


# ==============================================================================
# 4. PRUEBAS DE consultar_modelo (Traducción de errores para el consumidor)
# ==============================================================================

def test_consultar_modelo_convierte_error():
    """Confirma que al agotarse los reintentos, consultar_modelo traduzca el fallo en RuntimeError."""
    error_agotado = TimeoutError("Proveedor no disponible")

    with patch("cliente.con_reintento", side_effect=error_agotado):
        with pytest.raises(RuntimeError, match="El servicio no respondió tras varios intentos. Intenta más tarde."):
            consultar_modelo("Pedido de 5 sacos de harina panadera")
