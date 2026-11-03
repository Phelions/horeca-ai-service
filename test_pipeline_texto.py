"""
test_pipeline_texto.py - Suite de pruebas unitarias para el pipeline de texto y métricas (U2.2).

Verifica la normalización, tokenización, vectorización con Counter, y cálculo de
precisión, cobertura y F1 sin dependencias externas ni llamadas a red.
"""

import pytest
from collections import Counter

from pipeline_texto import (
    limpiar_texto,
    tokenizar,
    procesar_texto,
    vectorizar,
    contar_coincidencias,
    calcular_precision_cobertura_f1,
    calcular_metricas,
    resumir_metricas,
)


# ==============================================================================
# 1. PRUEBAS DE LIMPIEZA Y TOKENIZACIÓN (Bloque 15)
# ==============================================================================

def test_procesar_texto_normaliza_mayusculas_y_puntuacion():
    """Prueba fundamental del Bloque 15: mayúsculas y puntuación se unifican."""
    resultado = procesar_texto("El Cielo, es AZUL!")
    assert resultado == ["el", "cielo", "es", "azul"]


def test_procesar_texto_solo_puntuacion_devuelve_lista_vacia():
    """Prueba de caso borde del Bloque 15: texto solo de signos devuelve lista vacía."""
    resultado = procesar_texto("¡¡¡...!!!")
    assert resultado == []


def test_limpiar_texto_tipo_invalido_lanza_type_error():
    """Valida que entradas no-string lancen TypeError descriptivo."""
    with pytest.raises(TypeError, match="limpiar_texto espera str, no int"):
        limpiar_texto(12345)

    with pytest.raises(TypeError, match="limpiar_texto espera str, no NoneType"):
        limpiar_texto(None)


def test_limpiar_texto_preserva_tildes_y_caracteres_espanol():
    """Comprueba que no se pierdan tildes ni la 'ñ' en términos gastronómicos."""
    texto = "¡Ñoquis con Champiñones y Puré de Papas!"
    limpio = limpiar_texto(texto)
    assert limpio == "ñoquis con champiñones y puré de papas"


# ==============================================================================
# 2. PRUEBAS DE VECTORIZACIÓN Y CONTEO DE COINCIDENCIAS (Bloque 16)
# ==============================================================================

def test_vectorizar_cuenta_repeticiones_correctamente():
    """Verifica que la bolsa de palabras cuente la frecuencia exacta de cada token."""
    tokens = ["tomate", "cebolla", "tomate", "harina", "tomate"]
    conteo = vectorizar(tokens)
    assert conteo["tomate"] == 3
    assert conteo["cebolla"] == 1
    assert conteo["harina"] == 1


def test_contar_coincidencias_distingue_repeticiones():
    """
    Verifica que 'Counter & Counter' no se comporte como 'set()':
    si una palabra aparece 2 veces en la referencia y 4 en la respuesta, solo cuenta 2.
    """
    ref = ["el", "gato"]
    resp = ["el", "gato", "el", "gato"]
    coincidencias = contar_coincidencias(ref, resp)
    assert coincidencias == 2  # Min(1, 2) para 'el' + Min(1, 2) para 'gato'


# ==============================================================================
# 3. PRUEBAS DE MÉTRICAS: PRECISIÓN, COBERTURA Y F1 (Bloque 16)
# ==============================================================================

def test_calcular_precision_cobertura_f1_caso_perfecto():
    """Dos listas idénticas deben arrojar 1.0 en todas las métricas."""
    tokens = ["2", "cajas", "de", "tomate"]
    metricas = calcular_precision_cobertura_f1(tokens, tokens)
    assert metricas["precision"] == 1.0
    assert metricas["cobertura"] == 1.0
    assert metricas["f1"] == 1.0


def test_calcular_precision_cobertura_f1_listas_vacias():
    """Listas vacías devuelven 0.0 de forma segura sin ZeroDivisionError."""
    metricas = calcular_precision_cobertura_f1([], ["tomate"])
    assert metricas == {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}

    metricas_ambas = calcular_precision_cobertura_f1([], [])
    assert metricas_ambas == {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}


def test_calcular_precision_cobertura_f1_ejemplo_clase():
    """
    Verifica el cálculo del ejemplo paso a paso de la lección:
    Referencia: 6 tokens, Respuesta: 9 tokens, 5 coincidencias.
    Precision = 5/9 (~0.5556), Cobertura = 5/6 (~0.8333), F1 = ~0.6667.
    """
    ref = ["el", "gato", "duerme", "en", "el", "sofa"]
    resp = ["el", "gato", "duerme", "en", "el", "sillon", "todo", "el", "dia"]
    metricas = calcular_precision_cobertura_f1(ref, resp)

    assert pytest.approx(metricas["precision"], 0.001) == 5 / 9
    assert pytest.approx(metricas["cobertura"], 0.001) == 5 / 6
    esperado_f1 = 2 * (5 / 9) * (5 / 6) / ((5 / 9) + (5 / 6))
    assert pytest.approx(metricas["f1"], 0.001) == esperado_f1


def test_calcular_metricas_integrado():
    """Valida la función integradora de dos cadenas de texto completas."""
    ref = "2 cajas de tomate limachino"
    resp = "Tomate limachino: 2 cajas recibidas"
    metricas = calcular_metricas(ref, resp)
    # Tokens ref: ['2', 'cajas', 'de', 'tomate', 'limachino'] (5)
    # Tokens resp: ['tomate', 'limachino', '2', 'cajas', 'recibidas'] (5)
    # Coincidencias: '2', 'cajas', 'tomate', 'limachino' (4)
    assert metricas["precision"] == 0.8
    assert metricas["cobertura"] == 0.8
    assert metricas["f1"] == 0.8


def test_resumir_metricas_promedios():
    """Verifica el cálculo de promedios agregados sobre una lista de evaluaciones."""
    lista = [
        {"precision": 1.0, "cobertura": 0.5, "f1": 0.6667},
        {"precision": 0.6, "cobertura": 0.9, "f1": 0.7200},
    ]
    resumen = resumir_metricas(lista)
    assert resumen["precision_promedio"] == 0.8
    assert resumen["cobertura_promedio"] == 0.7
    assert resumen["f1_promedio"] == round((0.6667 + 0.7200) / 2, 4)


def test_resumir_metricas_lista_vacia():
    """Verifica que resumir una lista vacía devuelva promedios en 0.0."""
    resumen = resumir_metricas([])
    assert resumen == {
        "precision_promedio": 0.0,
        "cobertura_promedio": 0.0,
        "f1_promedio": 0.0,
    }

