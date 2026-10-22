"""
================================================================================
SUITE DE PRUEBAS AUTOMATIZADAS: test_order_parser.py
PROPÓSITO: Verificación y control de calidad unitario para Gate 1 con pytest.
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Ejecuta 12 pruebas automatizadas sin gastar saldo en Google ni depender de internet
(usando simulaciones o 'mocks' cuando corresponde). Garantiza que el código no tenga
regresiones y cumpla con todos los criterios críticos no compensables del Gate 1.

¿QUÉ CASOS SE PRUEBAN?
----------------------
1. Validación Temprana: Rechazar mensajes vacíos o tipos erróneos antes de llamar a la API.
2. Cálculo de Costos: Comprobar la fórmula matemática de tokens y tarifas en USD.
3. Parseo y Limpieza: Confirmar que elimine bloques markdown (```json ... ```) sin fallar.
4. Esquemas Pydantic: Validar que los ítems del restaurante se tipen correctamente.
5. Manejo de Red: Simular caídas de internet (ConnectionError) y verificar que no bote el sistema.
6. Manejo de Errores HTTP: Simular claves inválidas (401) y límites de cuota (429).
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.order_parser import OrderParser, ItemPedido, PedidoExtraido
from src.consumo_api import (
    armar_peticion,
    extraer_texto,
    parsear_respuesta_json,
    calcular_costo,
    PRECIO_TOKEN_ENTRADA_DEFAULT,
    PRECIO_TOKEN_SALIDA_DEFAULT
)


# ==============================================================================
# 1. PRUEBAS DE VALIDACIÓN TEMPRANA (Evitar llamadas inútiles y caras)
# ==============================================================================

def test_order_parser_valida_texto_vacio():
    """Valida que un texto vacío o de puros espacios lance ValueError de inmediato."""
    os.environ["GEMINI_API_KEY"] = "fake_test_key"
    parser = OrderParser(model_name="gemini-3.8-flash")
    
    with pytest.raises(ValueError, match="no puede estar vacío"):
        parser.parsear_texto("")

    with pytest.raises(ValueError, match="no puede estar vacío"):
        parser.parsear_texto("   \n\t  ")


def test_order_parser_valida_tipo_invalido():
    """Valida que si alguien pasa un número o dato no-texto, lance ValueError."""
    os.environ["GEMINI_API_KEY"] = "fake_test_key"
    parser = OrderParser(model_name="gemini-3.8-flash")

    with pytest.raises(ValueError, match="debe ser texto"):
        parser.parsear_texto(12345)  # type: ignore


def test_armar_peticion_valida_argumentos_vacios():
    """Valida que armar_peticion rechace preguntas o instrucciones de sistema vacías."""
    with pytest.raises(ValueError, match="no puede estar vacía"):
        armar_peticion("", "Eres un asistente de compras")

    with pytest.raises(ValueError, match="no puede estar vacía"):
        armar_peticion("Hola", "")


# ==============================================================================
# 2. PRUEBAS DE CÁLCULO DE TOKENS Y COSTO ECONÓMICO (U1.4 Bloque 8)
# ==============================================================================

def test_calcular_costo_formula_exacta():
    """
    Comprueba que la fórmula matemática se aplique con precisión:
    costo = (tokens_entrada * precio_entrada) + (tokens_salida * precio_salida)
    """
    datos_simulados = {
        "usageMetadata": {
            "promptTokenCount": 850,
            "candidatesTokenCount": 320,
            "totalTokenCount": 1170
        }
    }
    
    # Precios de prueba
    p_in = 0.00000075
    p_out = 0.00000375

    resultado = calcular_costo(datos_simulados, precio_entrada=p_in, precio_salida=p_out)

    esperado_in = 850 * p_in                    # 0.0006375
    esperado_out = 320 * p_out                  # 0.0012000
    esperado_total = esperado_in + esperado_out # 0.0018375

    assert resultado["tokens_entrada"] == 850
    assert resultado["tokens_salida"] == 320
    assert resultado["tokens_total"] == 1170
    assert pytest.approx(resultado["costo_total_usd"], rel=1e-5) == esperado_total


def test_calcular_costo_tolerante_a_usage_metadata_vacio():
    """Si la respuesta no trae metadata de uso, responde 0 en lugar de romperse con KeyError."""
    resultado = calcular_costo({})
    assert resultado["tokens_entrada"] == 0
    assert resultado["tokens_salida"] == 0
    assert resultado["costo_total_usd"] == 0.0


# ==============================================================================
# 3. PRUEBAS DE LIMPIEZA Y PARSEO JSON (U1.5 Bloque 10)
# ==============================================================================

def test_parsear_respuesta_json_limpia_bloques_markdown():
    """Verifica que limpie los bloques '```json ... ```' y devuelva un diccionario real."""
    texto_con_markdown = """```json
    {
        "categoria": "lacteos",
        "confianza": 0.95
    }
    ```"""
    datos = parsear_respuesta_json(texto_con_markdown)
    assert isinstance(datos, dict)
    assert datos["categoria"] == "lacteos"
    assert datos["confianza"] == 0.95


def test_parsear_respuesta_json_detecta_json_corrupto():
    """Valida que si el modelo devuelve un JSON incompleto o roto, lance ValueError claro."""
    texto_roto = '{"categoria": "carnes", "confianza": '
    with pytest.raises(ValueError, match="no pudo interpretarse como JSON"):
        parsear_respuesta_json(texto_roto)


# ==============================================================================
# 4. PRUEBAS DE ESQUEMAS PYDANTIC (Dominio Gastronómico HORECA)
# ==============================================================================

def test_pydantic_item_pedido_valido():
    """Verifica que un insumo típico de restaurante se cree correctamente con sus tipos."""
    item = ItemPedido(
        producto_mencionado="tomate limachino",
        cantidad=2.5,
        unidad_mencionada="caja",
        observaciones="bien firme y maduro"
    )
    assert item.producto_mencionado == "tomate limachino"
    assert item.cantidad == 2.5
    assert item.unidad_mencionada == "caja"
    assert item.observaciones == "bien firme y maduro"


def test_pydantic_pedido_extraido_lista_vacia():
    """Comprueba que una lista de compras vacía sea válida en el esquema."""
    pedido = PedidoExtraido(items=[])
    assert len(pedido.items) == 0


# ==============================================================================
# 5. PRUEBAS DE RESILIENCIA Y MANEJO DE RED / HTTP (Sin internet real)
# ==============================================================================

@patch("src.consumo_api.requests.post")
def test_enviar_peticion_manejo_connection_error(mock_post):
    """Simula que el cable de red se desconectó y valida que no se caiga el programa."""
    from requests.exceptions import ConnectionError
    mock_post.side_effect = ConnectionError("Falla de DNS simulada")

    from src.consumo_api import enviar_peticion
    with pytest.raises(RuntimeError, match="Error de red"):
        enviar_peticion("https://servidor-ficticio.com", {"body": 1})


@patch("src.consumo_api.requests.post")
def test_enviar_peticion_manejo_error_429_cuota(mock_post):
    """Simula que Google respondió con HTTP 429 (límite de cuota) y da mensaje amigable."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Quota exceeded"
    mock_post.return_value = mock_resp

    from src.consumo_api import enviar_peticion
    with pytest.raises(RuntimeError, match="Cuota excedida"):
        enviar_peticion("https://servidor-ficticio.com", {"body": 1})


@patch("src.consumo_api.requests.post")
def test_enviar_peticion_manejo_error_401_autenticacion(mock_post):
    """Simula que la API Key es errónea (HTTP 401) y lanza PermissionError comprensible."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "API key not valid"
    mock_post.return_value = mock_resp

    from src.consumo_api import enviar_peticion
    with pytest.raises(PermissionError, match="Error de autenticación"):
        enviar_peticion("https://servidor-ficticio.com", {"body": 1})
