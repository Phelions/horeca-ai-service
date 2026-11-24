"""
test_u26.py - Suite de pruebas automatizadas para la Unidad 2.6 (Bloques 29 y 30).

Verifica:
1. verificar_entorno.py: clasificación de errores de API y verificación del SDK/entorno.
2. dataset_tuning.py: al menos 10 ejemplos HORECA reales, balance entre categorías, validación
   limpia (0 problemas) y detección real de ejemplos inválidos, config PEFT y polling mock.
3. decision.py: estimación de costos de etiquetado, función recomendar_estrategia reportando
   empates explícitamente, evaluación real de respuestas_horeca, análisis de sensibilidad y
   esqueleto de 4 secciones.
4. decision_personalizacion.md: existencia, 4 secciones en orden exacto, ausencia de marcadores
   pendientes [COMPLETAR] y presencia de cifras concretas en la sección 4.
"""

import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.genai import types

from dataset_tuning import (
    CLAVES_ESPERADAS_HORECA,
    chequear_balance,
    construir_configuracion_peft,
    ejemplos_raw,
    esperar_a_que_termine,
    examples,
    validar_dataset,
)
from decision import (
    PROBLEMA_HORECA,
    analizar_sensibilidad,
    estimar_costo_de_etiquetar,
    generar_esqueleto_decision,
    recomendar_estrategia,
    respuestas_horeca,
)
from verificar_entorno import explicar_error, verificar_entorno_tuning


def test_explicar_error_distingue_autenticacion_cuota_y_conexion():
    """Valida mensajes claros por tipo de error sin ocultarlos."""
    assert "autenticación" in explicar_error(Exception("401 UNAUTHENTICATED API key not valid"))
    assert "Cuota" in explicar_error(Exception("429 RESOURCE_EXHAUSTED"))
    assert "conexión" in explicar_error(Exception("Connection timed out"))
    assert "Error inesperado" in explicar_error(RuntimeError("fallo desconocido"))


def test_verificar_entorno_tuning_con_env_cargado(monkeypatch):
    """Confirma que verificar_entorno_tuning expone los tipos TuningMode y AdapterSize del SDK."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaDummyTestKeyU26")
    info = verificar_entorno_tuning()
    assert info["api_key_detectada"] is True
    assert any("TUNING_MODE_PEFT_ADAPTER" in modo for modo in info["modos_tuning"])
    assert any("ADAPTER_SIZE_FOUR" in tam for tam in info["tamanios_adapter"])
    assert "epoch_count" in info["campos_config"]


def test_dataset_tuning_cumple_minimo_balance_y_validacion():
    """El dataset propio de HORECA tiene >=10 ejemplos, balance uniforme y 0 errores."""
    assert len(examples) >= 10
    assert len(examples) == len(ejemplos_raw)

    balance = chequear_balance(ejemplos_raw, "categoria")
    assert len(balance) == 4
    assert max(balance.values()) == min(balance.values())

    problemas = validar_dataset(examples, claves_esperadas=CLAVES_ESPERADAS_HORECA)
    assert problemas == []


def test_validar_dataset_detecta_todos_los_errores_inyectados():
    """Confirma que validar_dataset reporta texto vacío, clave faltante, duplicado y JSON inválido."""
    rotos = examples + [
        types.TuningExample(text_input="", output='{"categoria": "pedido_compra"}'),
        types.TuningExample(text_input=ejemplos_raw[0][0], output="no es json"),
    ]
    problemas = validar_dataset(rotos, claves_esperadas=CLAVES_ESPERADAS_HORECA)
    assert len(problemas) == 4
    assert any("text_input vacio" in p for p in problemas)
    assert any('no tiene la clave "prioridad"' in p for p in problemas)
    assert any("text_input duplicado" in p for p in problemas)
    assert any("output no es JSON valido" in p for p in problemas)


def test_construir_config_peft_y_esperar_a_que_termine_con_mock(monkeypatch):
    """Valida creación de configuración PEFT y bucle de espera hasta estado SUCCEEDED."""
    monkeypatch.setattr("dataset_tuning.time.sleep", lambda _: None)
    dataset, config = construir_configuracion_peft()
    assert len(dataset.examples) == len(examples)
    assert config.tuning_mode == types.TuningMode.TUNING_MODE_PEFT_ADAPTER
    assert config.adapter_size == types.AdapterSize.ADAPTER_SIZE_FOUR

    client_mock = MagicMock()
    client_mock.tunings.get.side_effect = [
        SimpleNamespace(state=types.JobState.JOB_STATE_RUNNING),
        SimpleNamespace(state=types.JobState.JOB_STATE_SUCCEEDED),
    ]
    trabajo_final = esperar_a_que_termine(client_mock, "tunings/demo-1", intervalo_segundos=1)
    assert trabajo_final.state == types.JobState.JOB_STATE_SUCCEEDED
    assert client_mock.tunings.get.call_count == 2


def test_estimar_costo_de_etiquetar_referencia_500_ejemplos():
    """Comprueba la fórmula lineal y el mini-ejercicio de 500 ejemplos + 20% segunda revisión."""
    horas_base, costo_base = estimar_costo_de_etiquetar(500, 4, 20)
    horas_rev, costo_rev = estimar_costo_de_etiquetar(500 * 0.20, 2, 20)
    assert round(horas_base, 2) == 33.33
    assert round(costo_base, 2) == 666.67
    assert round(horas_rev, 2) == 3.33
    assert round(costo_rev, 2) == 66.67
    assert round(horas_base + horas_rev, 2) == 36.67
    assert round(costo_base + costo_rev, 2) == 733.33


def test_recomendar_estrategia_reporta_empates_y_ganador_horeca():
    """Verifica que empates queden visibles y que respuestas_horeca recomiende RAG con 5 puntos."""
    caso_empate = {
        "informacion_cambia_seguido": False,
        "problema_de_formato_o_tono": True,
        "dataset_grande_y_de_calidad": True,
        "presupuesto_limitado_o_iterar_rapido": False,
        "necesita_multiples_variantes": False,
    }
    puntos_emp, ganadoras_emp = recomendar_estrategia(caso_empate)
    assert puntos_emp == {"RAG": 0, "Fine-tuning completo": 3, "PEFT": 3}
    assert ganadoras_emp == ["Fine-tuning completo", "PEFT"]

    puntos_h, ganadoras_h = recomendar_estrategia(respuestas_horeca)
    assert puntos_h == {"RAG": 5, "Fine-tuning completo": 0, "PEFT": 1}
    assert ganadoras_h == ["RAG"]

    sensibilidad = analizar_sensibilidad(respuestas_horeca)
    assert len(sensibilidad) == 5
    fila_formato = next(f for f in sensibilidad if f["clave"] == "problema_de_formato_o_tono")
    assert fila_formato["puntos"] == {"RAG": 5, "Fine-tuning completo": 1, "PEFT": 3}


def test_generar_esqueleto_y_documento_final_4_secciones():
    """Valida esqueleto programático y documento final decision_personalizacion.md."""
    esqueleto_empate = generar_esqueleto_decision(
        PROBLEMA_HORECA,
        {"RAG": 0, "Fine-tuning completo": 3, "PEFT": 3},
        ["Fine-tuning completo", "PEFT"],
    )
    assert "**Empate entre: Fine-tuning completo, PEFT**" in esqueleto_empate

    ruta_doc = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "decision_personalizacion.md"
    )
    assert os.path.exists(ruta_doc)
    contenido = open(ruta_doc, encoding="utf-8").read()

    encabezados = re.findall(r"^##\s+(.*)", contenido, re.MULTILINE)
    assert encabezados == [
        "1. El problema",
        "2. Resultado del método de decisión",
        "3. Técnica elegida y por qué",
        "4. Qué necesitarías para ejecutarla",
    ]
    assert "[COMPLETAR" not in contenido
    assert "5 puntos" in contenido
    assert "188" in contenido and "50 documentos" in contenido
