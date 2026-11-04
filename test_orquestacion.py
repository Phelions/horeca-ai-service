"""
test_orquestacion.py - Pruebas unitarias y de extremo a extremo para el flujo orquestado (Bloque 20 / U2.3).

Verifica exhaustivamente el checklist de orquestación oficial:
1. test_clasificacion_normaliza_antes_de_comparar
2. test_enrutamiento_va_a_cadena_queja
3. test_enrutamiento_va_a_cadena_pregunta
4. test_enrutamiento_usa_default_ante_categoria_desconocida
5. test_cadena_queja_usa_fallback_ante_fallo_del_modelo
6. test_procesar_mensaje_seguro_no_propaga_excepciones
Y la integración con el pipeline multi-paso de pedidos en HORECA.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

# Imports oficiales exigidos por el Bloque 20 desde enrutamiento.py
from enrutamiento import (
    construir_paso_clasificacion,
    construir_cadena_con_enrutamiento,
    construir_cadena_queja_segura,
    procesar_mensaje_seguro,
)

# Imports de la orquestación de dominio HORECA
from orquestacion import (
    construir_flujo_orquestado,
    construir_cadena_pedido,
)


# ============================================================================
# CHECKLIST OFICIAL DE ORQUESTACIÓN (BLOQUE 20 - ACTIVIDAD CORE)
# ============================================================================

def test_clasificacion_normaliza_antes_de_comparar():
    """
    Verifica que el clasificador aplique .strip().lower() dejando la categoría
    en minúsculas y sin espacios, sin importar cómo la devuelva el modelo.
    """
    modelo_falso = MagicMock()
    modelo_falso.return_value = AIMessage(content=" Queja \n")
    resultado = construir_paso_clasificacion(modelo_falso).invoke(
        {"mensaje": "algo salió mal con el despacho"}
    )
    assert resultado == "queja"


def test_enrutamiento_va_a_cadena_queja():
    """
    Verifica que un mensaje clasificado como 'queja' ejecute cadena_queja
    y no cualquier otra ruta.
    """
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
    """
    Verifica que un mensaje clasificado como 'pregunta' ejecute cadena_pregunta.
    """
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.return_value = "pregunta"
    cadena_pregunta_falsa = MagicMock()
    cadena_pregunta_falsa.return_value = "respuesta de pregunta"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
        cadena_pregunta=cadena_pregunta_falsa,
    )
    resultado = cadena.invoke({"mensaje": "¿cómo consulto los precios?"})

    assert cadena_pregunta_falsa.called
    assert resultado == "respuesta de pregunta"


def test_enrutamiento_usa_default_ante_categoria_desconocida():
    """
    Verifica que una categoría no reconocida por ninguna condición caiga
    en la ruta por defecto sin lanzar ningún error.
    """
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.return_value = "categoria_inventada"
    cadena_pregunta_falsa = MagicMock()
    cadena_pregunta_falsa.return_value = "respuesta por defecto"

    cadena = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
        cadena_pregunta=cadena_pregunta_falsa,
    )
    resultado = cadena.invoke({"mensaje": "mensaje ambiguo"})

    assert cadena_pregunta_falsa.called
    assert resultado == "respuesta por defecto"


def test_cadena_queja_usa_fallback_ante_fallo_del_modelo():
    """
    Verifica que si la cadena de queja falla por caída de red/API,
    with_fallbacks ejecute la respuesta de respaldo sin propagar la excepción.
    """
    def fallar_a_proposito(x):
        raise ConnectionError("simulado")

    cadena_queja_falsa = RunnableLambda(fallar_a_proposito)
    cadena_queja_segura = construir_cadena_queja_segura(cadena_queja_falsa)

    resultado = cadena_queja_segura.invoke({"mensaje": "esto no funciona"})

    assert "equipo" in resultado.lower()


def test_procesar_mensaje_seguro_no_propaga_excepciones():
    """
    Verifica que ningún fallo en la cadena (incluso antes de elegir ruta)
    deje sin respuesta a quien llame a procesar_mensaje_seguro.
    """
    clasificacion_falsa = MagicMock()
    clasificacion_falsa.side_effect = ConnectionError("simulado")

    cadena_falsa = construir_cadena_con_enrutamiento(
        paso_clasificacion=clasificacion_falsa,
    )

    resultado = procesar_mensaje_seguro(cadena_falsa, "mensaje de prueba")

    assert "no pudimos procesar" in resultado.lower()


# ============================================================================
# PRUEBAS ADICIONALES DEL DOMINIO GASTRONÓMICO HORECA
# ============================================================================

def test_flujo_orquestado_ruta_pedido_ejecuta_tres_pasos():
    """
    Verifica que al clasificar como 'pedido', el orquestador derive hacia
    la cadena de pedidos y ejecute las 3 etapas secuenciales.
    """
    clasificador_mock = MagicMock()
    clasificador_mock.return_value = "pedido"

    modelo_pedido = MagicMock()
    modelo_pedido.side_effect = [
        AIMessage(content="tomate: 5 cajas\ncebolla: 3 sacos"),
        AIMessage(content="Tomate, 5, cajas\nCebolla, 3, sacos"),
        AIMessage(content="- 5 cajas Tomate Limachino\n- 3 sacos Cebolla"),
    ]

    cadena_ped = construir_cadena_pedido(modelo_primario=modelo_pedido)

    flujo = construir_flujo_orquestado(
        modelo_primario=modelo_pedido,
        paso_clasificacion=clasificador_mock,
        cadena_pedido=cadena_ped,
    )

    resultado = flujo.invoke({"mensaje": "Mándame 5 cajas de tomate y 3 sacos de cebolla"})

    assert clasificador_mock.called
    assert modelo_pedido.call_count == 3
    assert "- 5 cajas Tomate Limachino" in resultado


def test_flujo_orquestado_ruta_reclamo():
    """Verifica que un mensaje de reclamo sea derivado a la sub-cadena de reclamos."""
    clasificador_mock = MagicMock()
    clasificador_mock.return_value = "reclamo"

    cadena_reclamo_mock = MagicMock()
    cadena_reclamo_mock.return_value = "Lamentamos la inconsistencia, abrimos ticket de resolución."

    flujo = construir_flujo_orquestado(
        modelo_primario=MagicMock(),
        paso_clasificacion=clasificador_mock,
        cadena_reclamo=cadena_reclamo_mock,
    )

    resultado = flujo.invoke({"mensaje": "Nos cobraron de más en el despacho de verduras"})

    assert cadena_reclamo_mock.called
    assert "ticket de resolución" in resultado


def test_flujo_orquestado_ruta_por_defecto_ante_categoria_desconocida():
    """
    Verifica que si el clasificador devuelve una categoría no contemplada,
    la bifurcación caiga de forma segura en la ruta por defecto (consulta).
    """
    clasificador_mock = MagicMock()
    clasificador_mock.return_value = "conversacion_informal"

    cadena_consulta_mock = MagicMock()
    cadena_consulta_mock.return_value = "Respuesta asistida para consulta de inventario."

    flujo = construir_flujo_orquestado(
        modelo_primario=MagicMock(),
        paso_clasificacion=clasificador_mock,
        cadena_consulta=cadena_consulta_mock,
    )

    resultado = flujo.invoke({"mensaje": "¿tienen catálogo nuevo disponible?"})

    assert cadena_consulta_mock.called
    assert resultado == "Respuesta asistida para consulta de inventario."


def test_resiliencia_with_fallbacks_en_paso_de_extraccion():
    """
    Verifica el manejo de errores por paso: si el modelo primario lanza
    una excepción (ej: TimeoutError / rate limit), el fallback captura
    la falla y utiliza el modelo de respaldo sin tumbar el flujo.
    """
    modelo_primario = MagicMock()
    # Falla en el primer paso (extracción)
    modelo_primario.side_effect = TimeoutError("Proveedor Gemini ocupado (429/Timeout)")

    modelo_respaldo = MagicMock()
    modelo_respaldo.side_effect = [
        AIMessage(content="insumos extraídos vía modelo respaldo"),
        AIMessage(content="insumos normalizados vía modelo respaldo"),
        AIMessage(content="- orden final completada vía respaldo"),
    ]

    cadena_resiliente = construir_cadena_pedido(
        modelo_primario=modelo_primario,
        modelo_respaldo=modelo_respaldo,
    )

    resultado = cadena_resiliente.invoke({"mensaje": "3 bidones de lavaloza industrial"})

    assert modelo_primario.called
    assert modelo_respaldo.called
    assert "- orden final completada vía respaldo" in resultado


def test_fallback_heuristico_cuando_no_hay_segundo_modelo():
    """
    Verifica que si no hay un segundo modelo disponible, el fallback heurístico
    rescate el flujo permitiendo continuar sin excepción fatal.
    """
    modelo_primario = MagicMock()
    # Falla en el paso 1, pero paso 2 y paso 3 se ejecutan con éxito
    modelo_primario.side_effect = [
        TimeoutError("API no disponible"),
        AIMessage(content="normalizado contingencia"),
        AIMessage(content="- orden contingencia"),
    ]

    cadena = construir_cadena_pedido(modelo_primario=modelo_primario, modelo_respaldo=None)
    resultado = cadena.invoke({"mensaje": "10 kg de harina pastelera"})

    assert "orden contingencia" in resultado


def test_orquestador_requiere_clave_mensaje():
    """Verifica que la invocación sin la variable requerida arroje KeyError inmediatamente."""
    flujo = construir_flujo_orquestado(modelo_primario=MagicMock())

    with pytest.raises(KeyError):
        flujo.invoke({"texto_invalido": "hola"})


def test_normalizacion_de_clasificacion():
    """Verifica que la función del clasificador limpie comillas, saltos y mayúsculas."""
    modelo_mock = MagicMock()
    modelo_mock.return_value = AIMessage(content=" 'PEDIDO' \n")

    paso = construir_paso_clasificacion(modelo_mock)
    resultado = paso.invoke({"mensaje": "necesito carne picada"})

    assert resultado == "pedido"
