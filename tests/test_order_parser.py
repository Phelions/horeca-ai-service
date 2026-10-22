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


# ============================================================
# 1. Pruebas de Validación Temprana (Criterio de Calidad U1.3)
# ============================================================

def test_order_parser_valida_texto_vacio():
    """Valida que una entrada vacía o de solo espacios lance ValueError."""
    os.environ["GEMINI_API_KEY"] = "fake_test_key"
    parser = OrderParser(model_name="gemini-2.5-flash")
    
    with pytest.raises(ValueError, match="no puede estar vacío"):
        parser.parsear_texto("")

    with pytest.raises(ValueError, match="no puede estar vacío"):
        parser.parsear_texto("   \n\t  ")


def test_order_parser_valida_tipo_invalido():
    """Valida que un tipo de entrada no-string lance ValueError."""
    os.environ["GEMINI_API_KEY"] = "fake_test_key"
    parser = OrderParser(model_name="gemini-2.5-flash")

    with pytest.raises(ValueError, match="debe ser de tipo str"):
        parser.parsear_texto(12345)  # type: ignore


def test_armar_peticion_valida_argumentos_vacios():
    """Valida que armar_peticion rechace preguntas o instrucciones vacías."""
    with pytest.raises(ValueError, match="no puede estar vacía"):
        armar_peticion("", "Eres un asistente")

    with pytest.raises(ValueError, match="no puede estar vacía"):
        armar_peticion("Hola", "")


# ============================================================
# 2. Pruebas de Cálculo de Tokens y Costo (U1.4 Bloque 8)
# ============================================================

def test_calcular_costo_formula_exacta():
    """
    Verifica que el cálculo económico aplique exactamente la fórmula:
    costo = (promptTokenCount * precio_in) + (candidatesTokenCount * precio_out)
    """
    datos_simulados = {
        "usageMetadata": {
            "promptTokenCount": 850,
            "candidatesTokenCount": 320,
            "totalTokenCount": 1170
        }
    }
    
    # Precios de prueba
    p_in = 0.00000010
    p_out = 0.00000040

    resultado = calcular_costo(datos_simulados, precio_entrada=p_in, precio_salida=p_out)

    esperado_in = 850 * p_in          # 0.000085
    esperado_out = 320 * p_out        # 0.000128
    esperado_total = esperado_in + esperado_out  # 0.000213

    assert resultado["tokens_entrada"] == 850
    assert resultado["tokens_salida"] == 320
    assert resultado["tokens_total"] == 1170
    assert pytest.approx(resultado["costo_total_usd"], rel=1e-5) == esperado_total


def test_calcular_costo_tolerante_a_usage_metadata_vacio():
    """Si no viene usageMetadata, debe responder en 0 sin romperse."""
    resultado = calcular_costo({})
    assert resultado["tokens_entrada"] == 0
    assert resultado["tokens_salida"] == 0
    assert resultado["costo_total_usd"] == 0.0


# ============================================================
# 3. Pruebas de Limpieza y Parseo JSON (U1.5 Bloque 10)
# ============================================================

def test_parsear_respuesta_json_limpia_bloques_markdown():
    """Verifica que elimine ```json ... ``` y devuelva un diccionario."""
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
    """Valida que un JSON incompleto o roto lance ValueError explícito."""
    texto_roto = '{"categoria": "carnes", "confianza": '
    with pytest.raises(ValueError, match="no pudo interpretarse como JSON"):
        parsear_respuesta_json(texto_roto)


# ============================================================
# 4. Pruebas de Esquemas Pydantic para el Dominio HORECA
# ============================================================

def test_pydantic_item_pedido_valido():
    """Verifica la correcta instanciación y validación de tipos del ítem de compra."""
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
    """Una orden sin insumos válidos se representa con lista vacía conforme al esquema."""
    pedido = PedidoExtraido(items=[])
    assert len(pedido.items) == 0


# ============================================================
# 5. Pruebas de Manejo de Excepciones de Red y HTTP (U1.3)
# ============================================================

@patch("src.consumo_api.requests.post")
def test_enviar_peticion_manejo_connection_error(mock_post):
    """Simula falla física de red y verifica que lance RuntimeError descriptivo."""
    from requests.exceptions import ConnectionError
    mock_post.side_effect = ConnectionError("DNS failure")

    from src.consumo_api import enviar_peticion
    with pytest.raises(RuntimeError, match="Error de red"):
        enviar_peticion("https://fake-url.com", {"body": 1})


@patch("src.consumo_api.requests.post")
def test_enviar_peticion_manejo_error_429_cuota(mock_post):
    """Simula respuesta HTTP 429 (límite de cuota) y verifica mensaje claro."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Quota exceeded"
    mock_post.return_value = mock_resp

    from src.consumo_api import enviar_peticion
    with pytest.raises(RuntimeError, match="Cuota excedida"):
        enviar_peticion("https://fake-url.com", {"body": 1})


@patch("src.consumo_api.requests.post")
def test_enviar_peticion_manejo_error_401_autenticacion(mock_post):
    """Simula respuesta HTTP 401 (API key inválida) y verifica PermissionError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "API key not valid"
    mock_post.return_value = mock_resp

    from src.consumo_api import enviar_peticion
    with pytest.raises(PermissionError, match="Error de autenticación"):
        enviar_peticion("https://fake-url.com", {"body": 1})
