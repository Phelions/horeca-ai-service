"""
test_agente_con_herramientas.py - Suite de pruebas automatizadas con pytest (Unidad 2.4 / Bloques 21, 22 y 23).

Verifica exhaustivamente:
1. Selección de herramientas y argumentos en intermediate_steps.
2. Respuesta directa (intermediate_steps == []) en consultas generales.
3. Aislamiento de memoria entre sesiones con RunnableWithMessageHistory.
4. Manejo robusto de excepciones internas en herramientas (retorno str).
5. Detección de repeticiones consecutivas en marcar_repeticiones_sospechosas y manejo de FileNotFoundError.
6. Invocación de esperar_aprobacion y registrar_aprobacion exclusivamente en herramientas de riesgo alto.
7. Comportamiento seguro de esperar_aprobacion ante timeout o rechazo con select.select.
8. Validación estricta del orden de placeholders en ChatPromptTemplate.
"""

import json
import os
import re
import sys
from datetime import datetime
from decimal import Decimal
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from agente_con_herramientas import (
    HERRAMIENTAS_HORECA,
    NIVELES_RIESGO_HERRAMIENTAS,
    PROMPT_SISTEMA_HORECA,
    calcular_cotizacion,
    cancelar_borrador_pedido,
    construir_agente_con_historial,
    construir_agente_horeca,
    construir_ejecutor_agente,
    construir_modelo_llm,
    consultar_catalogo_insumos,
    crear_borrador_pedido,
    ejecutar_agente,
    emitir_orden_compra_formal,
    esperar_aprobacion,
    limpiar_historial_sesiones,
    limpiar_registros_transaccionales,
    marcar_repeticiones_sospechosas,
    obtener_historial_sesion,
    procesar_pago_proveedor,
    prompt_agente_horeca,
    registrar_aprobacion,
    registrar_traza,
)


@pytest.fixture(autouse=True)
def reiniciar_estado_aislado():
    """Garantiza aislamiento estricto entre pruebas reseteando historiales y estado transaccional."""
    limpiar_historial_sesiones()
    limpiar_registros_transaccionales()
    yield
    limpiar_historial_sesiones()
    limpiar_registros_transaccionales()


# ============================================================================
# CLASE DE MODELO CHAT MOCK DETERMINÍSTICO (SIN RED / APTO SANDBOX)
# ============================================================================

class ModeloChatMock(BaseChatModel):
    """Modelo de chat simulado para pruebas unitarias deterministas sin conexión externa."""

    respuestas: List[AIMessage] = []

    def _generate(
        self,
        messages: List[Any],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.respuestas:
            mensaje = self.respuestas.pop(0)
        else:
            mensaje = AIMessage(content="Respuesta simulada por defecto")
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    @property
    def _llm_type(self) -> str:
        return "modelo-chat-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        # Permite la compatibilidad con create_tool_calling_agent
        return self


# ============================================================================
# 1. PRUEBAS DE SELECCIÓN DE HERRAMIENTAS Y ARGUMENTOS EN INTERMEDIATE_STEPS
# ============================================================================

def test_seleccion_herramienta_y_argumentos_en_intermediate_steps(tmp_path):
    """Verifica que el agente invoque la herramienta adecuada, pase los argumentos correctos

    y los registre en intermediate_steps.
    """
    ruta_trazas = str(tmp_path / "trazas_test.jsonl")

    # Paso 1: El LLM decide llamar a consultar_catalogo_insumos con nombre_insumo="tomate triturado"
    # Paso 2: Tras recibir la observación, el LLM emite la respuesta final
    respuestas = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "consultar_catalogo_insumos",
                    "args": {"nombre_insumo": "tomate triturado"},
                    "id": "llamada_herramienta_01",
                }
            ],
        ),
        AIMessage(content="Tenemos 80 latas de tomate triturado en bodega."),
    ]
    modelo_mock = ModeloChatMock(respuestas=respuestas)
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente, return_intermediate_steps=True)

    with patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas):
        resultado = ejecutor.invoke({
            "entrada": "¿Cuánto stock nos queda de tomate triturado?",
            "chat_history": [],
        })

    # Verificaciones sobre intermediate_steps
    assert "intermediate_steps" in resultado
    pasos = resultado["intermediate_steps"]
    assert len(pasos) == 1

    accion_agente, observacion = pasos[0]
    assert accion_agente.tool == "consultar_catalogo_insumos"
    assert accion_agente.tool_input == {"nombre_insumo": "tomate triturado"}
    assert "80.0 latas" in observacion or "80" in observacion
    assert "Agrícola El Sol" in observacion

    # Verificación de la respuesta final del agente
    assert "Tenemos 80 latas" in resultado["output"]


def test_seleccion_herramienta_con_esquema_pydantic_cotizacion(tmp_path):
    """Verifica la invocación de herramientas con esquema Pydantic (calcular_cotizacion con > 1 argumento)."""
    ruta_trazas = str(tmp_path / "trazas_test_cotizacion.jsonl")

    respuestas = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calcular_cotizacion",
                    "args": {"nombre_insumo": "harina", "cantidad_unidades": 50.0},
                    "id": "llamada_cotizacion_01",
                }
            ],
        ),
        AIMessage(content="El costo total estimado para 50 kg de harina es $50,575.00 CLP con IVA."),
    ]
    modelo_mock = ModeloChatMock(respuestas=respuestas)
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente, return_intermediate_steps=True)

    with patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas):
        resultado = ejecutor.invoke({
            "entrada": "Cotízame 50 kg de harina de trigo",
            "chat_history": [],
        })

    pasos = resultado["intermediate_steps"]
    assert len(pasos) == 1
    accion, observacion = pasos[0]
    assert accion.tool == "calcular_cotizacion"
    assert accion.tool_input["nombre_insumo"] == "harina"
    assert float(accion.tool_input["cantidad_unidades"]) == 50.0
    assert "Cotización estimada para 'Harina de trigo fuerza 000'" in observacion
    assert "IVA (19%)" in observacion


# ============================================================================
# 2. PRUEBAS DE RESPUESTA DIRECTA SIN HERRAMIENTAS EN CONSULTAS GENERALES
# ============================================================================

def test_respuesta_directa_sin_herramientas_en_consulta_general():
    """Verifica que ante saludos o consultas generales el agente responda directamente

    sin invocar herramientas (intermediate_steps == []).
    """
    respuestas = [
        AIMessage(
            content="¡Hola! Soy tu asistente de compras HORECA. ¿En qué puedo ayudarte hoy en la cocina?"
        )
    ]
    modelo_mock = ModeloChatMock(respuestas=respuestas)
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente, return_intermediate_steps=True)

    resultado = ejecutor.invoke({
        "entrada": "Hola, buenas tardes, ¿cómo estás?",
        "chat_history": [],
    })

    assert "intermediate_steps" in resultado
    assert resultado["intermediate_steps"] == []
    assert "asistente de compras HORECA" in resultado["output"]


# ============================================================================
# 3. PRUEBAS DE AISLAMIENTO DE MEMORIA ENTRE SESIONES
# ============================================================================

def test_aislamiento_estricto_de_memoria_entre_sesiones():
    """Verifica que RunnableWithMessageHistory garantice el aislamiento estricto

    de historiales conversacionales por session_id sin contaminación cruzada.
    """
    limpiar_historial_sesiones()

    modelo_mock = ModeloChatMock(respuestas=[
        AIMessage(content="Entendido Chef Carlos, configurando compras para Restaurante Bellavista."),
        AIMessage(content="Entendido Chef Andrea, configurando compras para Café Costanera."),
    ])
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente)
    agente_con_historial = construir_agente_con_historial(ejecutor=ejecutor)

    # Invocación en Sesión 1: Restaurante Bellavista
    agente_con_historial.invoke(
        {"entrada": "Hola, soy el Chef Carlos de Restaurante Bellavista"},
        config={"configurable": {"session_id": "sesion_bellavista_01"}},
    )

    # Invocación en Sesión 2: Café Costanera
    agente_con_historial.invoke(
        {"entrada": "Hola, soy la Chef Andrea de Café Costanera"},
        config={"configurable": {"session_id": "sesion_costanera_02"}},
    )

    historial_bellavista = obtener_historial_sesion("sesion_bellavista_01")
    historial_costanera = obtener_historial_sesion("sesion_costanera_02")

    # Ambas sesiones deben tener exactamente 2 mensajes (1 HumanMessage + 1 AIMessage)
    assert len(historial_bellavista.messages) == 2
    assert len(historial_costanera.messages) == 2

    # Verificar que los datos de Bellavista no se hayan filtrado a Costanera
    mensajes_bellavista_texto = [m.content for m in historial_bellavista.messages]
    mensajes_costanera_texto = [m.content for m in historial_costanera.messages]

    assert any("Chef Carlos" in m for m in mensajes_bellavista_texto)
    assert any("Restaurante Bellavista" in m for m in mensajes_bellavista_texto)
    assert not any("Chef Carlos" in m for m in mensajes_costanera_texto)
    assert not any("Restaurante Bellavista" in m for m in mensajes_costanera_texto)

    assert any("Chef Andrea" in m for m in mensajes_costanera_texto)
    assert any("Café Costanera" in m for m in mensajes_costanera_texto)
    assert not any("Chef Andrea" in m for m in mensajes_bellavista_texto)
    assert not any("Café Costanera" in m for m in mensajes_bellavista_texto)


# ============================================================================
# 4. PRUEBAS DE MANEJO DE EXCEPCIONES EN HERRAMIENTAS (TRY/EXCEPT -> STR)
# ============================================================================

def test_herramientas_capturan_excepciones_y_devuelven_str_explicativo(tmp_path):
    """Verifica que todas las herramientas capturen fallos internos y devuelvan cadenas

    de tipo str explicativas sin propagar excepciones que aborten el ciclo del agente.
    """
    ruta_trazas = str(tmp_path / "trazas_errores.jsonl")

    with patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas):
        # 1. Entrada vacía en consultar_catalogo_insumos
        obs_1 = consultar_catalogo_insumos.invoke({"nombre_insumo": ""})
        assert isinstance(obs_1, str)
        assert "Error: El nombre del insumo no puede estar vacío" in obs_1

        # 2. Cantidad inválida (negativa) en calcular_cotizacion
        obs_2 = calcular_cotizacion.invoke({"nombre_insumo": "harina", "cantidad_unidades": -10.0})
        assert isinstance(obs_2, str)
        assert "Error de validación" in obs_2

        # 3. Cantidad cero en crear_borrador_pedido
        obs_3 = crear_borrador_pedido.invoke({
            "proveedor": "Molino Central",
            "nombre_insumo": "harina",
            "cantidad_unidades": 0.0,
        })
        assert isinstance(obs_3, str)
        assert "Error de validación" in obs_3

        # 4. Cancelación de borrador inexistente
        obs_4 = cancelar_borrador_pedido.invoke({"id_borrador": "BORR-NO-EXISTE-9999"})
        assert isinstance(obs_4, str)
        assert "No se encontró el borrador" in obs_4

        # 5. Simulación de fallo imprevisto interno
        with patch.dict("agente_con_herramientas.CATALOGO_INSUMOS_HORECA", {}, clear=True):
            # Insumo no encontrado en catálogo vacío
            obs_5 = consultar_catalogo_insumos.invoke({"nombre_insumo": "azafrán"})
            assert isinstance(obs_5, str)
            assert "No se encontró el insumo 'azafrán'" in obs_5


# ============================================================================
# 5. PRUEBAS DE DETECCIÓN DE REPETICIONES SOSPECHOSAS Y FILE NOT FOUND
# ============================================================================

def test_marcar_repeticiones_sospechosas_archivo_inexistente():
    """Verifica que marcar_repeticiones_sospechosas maneje limpiamente FileNotFoundError

    devolviendo una lista vacía sin arrojar excepciones.
    """
    resultado = marcar_repeticiones_sospechosas("archivo_inexistente_totalmente_ficticio.jsonl")
    assert isinstance(resultado, list)
    assert resultado == []


def test_marcar_repeticiones_sospechosas_detecta_llamadas_consecutivas(tmp_path):
    """Verifica que se detecten alertas cuando en una sesión la misma herramienta

    sea invocada de forma consecutiva (herramientas[i] == herramientas[i - 1]).
    """
    archivo_trazas = tmp_path / "trazas_sospechosas.jsonl"

    eventos = [
        # Sesión A: Dos llamadas consecutivas a consultar_catalogo_insumos -> ALERTA
        {
            "session_id": "sesion_A",
            "herramienta": "consultar_catalogo_insumos",
            "observacion": "Stock 150 kg",
        },
        {
            "session_id": "sesion_A",
            "herramienta": "consultar_catalogo_insumos",
            "observacion": "Stock 150 kg",
        },
        # Sesión A: Llamada a herramienta distinta -> NO ALERTA
        {
            "session_id": "sesion_A",
            "herramienta": "calcular_cotizacion",
            "observacion": "Total $20,000",
        },
        # Sesión B: Herramientas intercaladas -> NO ALERTA
        {
            "session_id": "sesion_B",
            "herramienta": "consultar_catalogo_insumos",
            "observacion": "Stock 60 L",
        },
        {
            "session_id": "sesion_B",
            "herramienta": "crear_borrador_pedido",
            "observacion": "BORR-0001 creado",
        },
        {
            "session_id": "sesion_B",
            "herramienta": "consultar_catalogo_insumos",
            "observacion": "Stock 60 L",
        },
        # Sesión C: Tres llamadas consecutivas a crear_borrador_pedido -> 2 ALERTAS
        {
            "session_id": "sesion_C",
            "herramienta": "crear_borrador_pedido",
            "observacion": "BORR-0002",
        },
        {
            "session_id": "sesion_C",
            "herramienta": "crear_borrador_pedido",
            "observacion": "BORR-0003",
        },
        {
            "session_id": "sesion_C",
            "herramienta": "crear_borrador_pedido",
            "observacion": "BORR-0004",
        },
    ]

    with open(archivo_trazas, "w", encoding="utf-8") as f:
        for ev in eventos:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    alertas = marcar_repeticiones_sospechosas(str(archivo_trazas))

    # Debe detectar 1 alerta en sesion_A y 2 alertas en sesion_C
    assert len(alertas) == 3

    sesiones_alertadas = [a["session_id"] for a in alertas]
    assert sesiones_alertadas.count("sesion_A") == 1
    assert sesiones_alertadas.count("sesion_B") == 0
    assert sesiones_alertadas.count("sesion_C") == 2

    # Verificar estructura de los metadatos de la alerta
    primera_alerta = alertas[0]
    assert primera_alerta["herramienta"] == "consultar_catalogo_insumos"
    assert "Invocación consecutiva sospechosa" in primera_alerta["mensaje"]


# ============================================================================
# 6. PRUEBAS DE INVOCACIÓN DE ESPERAR_APROBACION EXCLUSIVAMENTE EN RIESGO ALTO
# ============================================================================

def test_esperar_aprobacion_invocado_exclusivamente_en_riesgo_alto(tmp_path):
    """Verifica que esperar_aprobacion se ejecute única y exclusivamente

    en las herramientas de riesgo alto, y jamás en las de riesgo bajo o medio.
    """
    ruta_trazas = str(tmp_path / "trazas_hitl.jsonl")
    ruta_aprobaciones = str(tmp_path / "aprobaciones_hitl.jsonl")

    with (
        patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas),
        patch("agente_con_herramientas.ARCHIVO_APROBACIONES_DEFECTO", ruta_aprobaciones),
        patch("agente_con_herramientas.esperar_aprobacion") as mock_esperar,
    ):
        mock_esperar.return_value = True

        # 1. RIESGO BAJO: consultar_catalogo_insumos
        consultar_catalogo_insumos.invoke({"nombre_insumo": "harina"})
        mock_esperar.assert_not_called()

        # 2. RIESGO BAJO: calcular_cotizacion
        calcular_cotizacion.invoke({"nombre_insumo": "harina", "cantidad_unidades": 10.0})
        mock_esperar.assert_not_called()

        # 3. RIESGO MEDIO: crear_borrador_pedido
        crear_borrador_pedido.invoke({
            "proveedor": "Molino Central",
            "nombre_insumo": "harina",
            "cantidad_unidades": 10.0,
        })
        mock_esperar.assert_not_called()

        # 4. RIESGO MEDIO: cancelar_borrador_pedido
        cancelar_borrador_pedido.invoke({"id_borrador": "BORR-0001"})
        mock_esperar.assert_not_called()

        # 5. RIESGO ALTO: emitir_orden_compra_formal -> DEBE INVOCAR esperar_aprobacion
        obs_oc = emitir_orden_compra_formal.invoke({
            "id_borrador": "BORR-0001",
            "proveedor": "Molino Central",
            "monto_total_pesos": 150000.0,
        })
        assert mock_esperar.call_count == 1
        assert "ORDEN DE COMPRA EMITIDA CON ÉXITO" in obs_oc

        # 6. RIESGO ALTO: procesar_pago_proveedor -> DEBE INVOCAR esperar_aprobacion
        obs_pago = procesar_pago_proveedor.invoke({
            "proveedor": "Molino Central",
            "monto_pesos": 150000.0,
            "numero_factura": "FAC-8899",
        })
        assert mock_esperar.call_count == 2
        assert "TRANSFERENCIA BANCARIA COMPLETADA EXITOSAMENTE" in obs_pago


def test_esperar_aprobacion_rechazo_cancela_operacion_alto_riesgo(tmp_path):
    """Verifica que si esperar_aprobacion retorna False (rechazo u omisión),

    la operación de alto riesgo se cancele sin formalizar la orden ni transferir fondos.
    """
    ruta_trazas = str(tmp_path / "trazas_rechazo.jsonl")
    ruta_aprobaciones = str(tmp_path / "aprobaciones_rechazo.jsonl")

    with (
        patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas),
        patch("agente_con_herramientas.ARCHIVO_APROBACIONES_DEFECTO", ruta_aprobaciones),
        patch("agente_con_herramientas.esperar_aprobacion") as mock_esperar,
    ):
        mock_esperar.return_value = False

        obs_oc = emitir_orden_compra_formal.invoke({
            "id_borrador": "BORR-0002",
            "proveedor": "Distribuidora Valle",
            "monto_total_pesos": 450000.0,
        })
        assert mock_esperar.call_count == 1
        assert "OPERACIÓN CANCELADA" in obs_oc
        assert "fue rechazada por el operador humano" in obs_oc

        obs_pago = procesar_pago_proveedor.invoke({
            "proveedor": "Distribuidora Valle",
            "monto_pesos": 450000.0,
            "numero_factura": "FAC-9911",
        })
        assert mock_esperar.call_count == 2
        assert "PAGO CANCELADO" in obs_pago


def test_grep_esperar_aprobacion_coincide_unicamente_con_definicion_y_riesgo_alto():
    """Verifica el criterio formal del checklist mediante grep en agente_con_herramientas.py:

    coincide únicamente con la definición de la función y las llamadas en riesgo alto.
    """
    ruta_modulo = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "agente_con_herramientas.py"
    )
    with open(ruta_modulo, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    coincidencias = []
    for num_linea, linea in enumerate(lineas, start=1):
        if "esperar_aprobacion(" in linea:
            coincidencias.append((num_linea, linea.strip()))

    # Debe haber exactamente 3 coincidencias:
    # 1. Definición: def esperar_aprobacion(mensaje: str, ...)
    # 2. Invocación en emitir_orden_compra_formal
    # 3. Invocación en procesar_pago_proveedor
    assert len(coincidencias) == 3, f"Coincidencias encontradas: {coincidencias}"

    # Validar que la primera sea la definición de la función
    assert "def esperar_aprobacion(" in coincidencias[0][1]

    # Validar que las otras dos sean llamadas de asignación
    assert "autorizado = esperar_aprobacion(" in coincidencias[1][1]
    assert "autorizado = esperar_aprobacion(" in coincidencias[2][1]


# ============================================================================
# 7. PRUEBAS DE SELECT.SELECT EN ESPERAR_APROBACION (TIMEOUT Y SEGURIDAD)
# ============================================================================

def test_esperar_aprobacion_timeout_retorna_false_de_forma_segura():
    """Verifica que esperar_aprobacion retorne False cuando select.select expira por timeout."""
    with patch("select.select") as mock_select:
        # Simulamos que select devuelve lista vacía de lectores (timeout expirado)
        mock_select.return_value = ([], [], [])

        resultado = esperar_aprobacion(
            "Emisión formal de orden de compra", tiempo_limite_segundos=1
        )
        assert resultado is False


def test_esperar_aprobacion_respuestas_afirmativas_y_negativas():
    """Verifica que esperar_aprobacion reconozca entradas afirmativas ('s', 'si', 'y')

    y rechace entradas negativas ('n', 'no') o texto arbitrario.
    """
    with patch("select.select") as mock_select:
        mock_select.return_value = ([sys.stdin], [], [])

        # Caso afirmativo: 's'
        with patch.object(sys.stdin, "readline", return_value="s\n"):
            assert esperar_aprobacion("Test afirmativo 1") is True

        # Caso afirmativo: 'sí'
        with patch.object(sys.stdin, "readline", return_value="sí\n"):
            assert esperar_aprobacion("Test afirmativo 2") is True

        # Caso afirmativo: 'yes'
        with patch.object(sys.stdin, "readline", return_value="yes\n"):
            assert esperar_aprobacion("Test afirmativo 3") is True

        # Caso negativo: 'n'
        with patch.object(sys.stdin, "readline", return_value="n\n"):
            assert esperar_aprobacion("Test negativo 1") is False

        # Caso negativo: respuesta inesperada
        with patch.object(sys.stdin, "readline", return_value="quizas\n"):
            assert esperar_aprobacion("Test negativo 2") is False


# ============================================================================
# 8. PRUEBA DE ORDEN ESTRICTO DE PLACEHOLDERS EN CHATPROMPTTEMPLATE (R2)
# ============================================================================

def test_orden_estricto_placeholders_en_prompt_template():
    """Verifica el orden estricto exigido por R2 en ChatPromptTemplate:

    (1) 'system', (2) chat_history, (3) human {entrada}, (4) agent_scratchpad.
    """
    mensajes = prompt_agente_horeca.messages
    assert len(mensajes) == 4, f"Se esperaban exactamente 4 elementos, se hallaron {len(mensajes)}"

    # 1. Elemento del sistema ("system")
    assert isinstance(mensajes[0], SystemMessagePromptTemplate)

    # 2. MessagesPlaceholder("chat_history")
    assert isinstance(mensajes[1], MessagesPlaceholder)
    assert mensajes[1].variable_name == "chat_history"

    # 3. ("human", "{entrada}")
    assert isinstance(mensajes[2], HumanMessagePromptTemplate)
    assert "{entrada}" in mensajes[2].prompt.template

    # 4. MessagesPlaceholder("agent_scratchpad")
    assert isinstance(mensajes[3], MessagesPlaceholder)
    assert mensajes[3].variable_name == "agent_scratchpad"


# ============================================================================
# 9. PRUEBA DE PERSISTENCIA Y FORMATO UTF-8 EN REGISTRAR_TRAZA Y APROBACIONES
# ============================================================================

def test_registrar_traza_y_aprobacion_formato_utf8_jsonl(tmp_path):
    """Verifica que registrar_traza y registrar_aprobacion almacenen correctamente

    en codificación UTF-8 JSON Lines con caracteres especiales del español (tildes, eñes).
    """
    archivo_trazas = tmp_path / "trazas_utf8.jsonl"
    archivo_aprobaciones = tmp_path / "aprobaciones_utf8.jsonl"

    # Registrar traza con caracteres especiales del dominio gastronómico
    evento_traza = registrar_traza(
        session_id="sesion_gastronomica_ñandu",
        herramienta="consultar_catalogo_insumos",
        entrada={"nombre_insumo": "champiñón París y piña"},
        observacion="Stock verificado con éxito: 15 cajones de champiñón fresco.",
        ruta_archivo=str(archivo_trazas),
    )

    # Registrar aprobación
    evento_aprobacion = registrar_aprobacion(
        accion="emitir_orden_compra_formal",
        aprobada=True,
        usuario="auditor_gastronómico",
        detalles={"observación": "Aprobación autorizada para ración de menú"},
        ruta_archivo=str(archivo_aprobaciones),
    )

    # Leer archivos en bytes y validar UTF-8
    with open(archivo_trazas, "r", encoding="utf-8") as f:
        linea_traza = f.readline()
        datos_traza = json.loads(linea_traza)
        assert datos_traza["session_id"] == "sesion_gastronomica_ñandu"
        assert "champiñón" in datos_traza["entrada"]["nombre_insumo"]

    with open(archivo_aprobaciones, "r", encoding="utf-8") as f:
        linea_aprob = f.readline()
        datos_aprob = json.loads(linea_aprob)
        assert datos_aprob["usuario"] == "auditor_gastronómico"
        assert datos_aprob["aprobada"] is True


# ============================================================================
# 10. PRUEBA DE EJECUTAR_AGENTE CON CONTEXTO DE SESIÓN Y TRAZAS
# ============================================================================

def test_ejecutar_agente_propaga_session_id_a_trazas(tmp_path):
    """Verifica que la función ejecutar_agente preserve y propague el session_id

    a las herramientas invocadas y a las trazas JSON Lines registradas.
    """
    ruta_trazas = str(tmp_path / "trazas_ejecucion_sesion.jsonl")

    respuestas = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "consultar_catalogo_insumos",
                    "args": {"nombre_insumo": "aceite de oliva"},
                    "id": "llamada_aceite_01",
                }
            ],
        ),
        AIMessage(content="El aceite de oliva tiene 60 L en inventario."),
    ]
    modelo_mock = ModeloChatMock(respuestas=respuestas)
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente, return_intermediate_steps=True)
    agente_runnable = construir_agente_con_historial(ejecutor=ejecutor)

    with patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas):
        resultado = ejecutar_agente(
            entrada="¿Cuánto aceite de oliva tenemos?",
            session_id="sesion_cocina_central_09",
            agente_ejecutable=agente_runnable,
        )

    assert "60 L" in resultado["output"]
    assert len(resultado["intermediate_steps"]) == 1

    # Verificar que el session_id haya quedado registrado en la traza
    with open(ruta_trazas, "r", encoding="utf-8") as f:
        lineas = f.readlines()
        assert len(lineas) >= 1
        traza = json.loads(lineas[0])
        assert traza["session_id"] == "sesion_cocina_central_09"
        assert traza["herramienta"] == "consultar_catalogo_insumos"


# ============================================================================
# 11. PRUEBA DE RESILIENCIA EN MARCAR_REPETICIONES_SOSPECHOSAS (NO-DICT Y CORRUPTO)
# ============================================================================

def test_marcar_repeticiones_sospechosas_tolera_lineas_no_dict_y_corruptas(tmp_path):
    """Verifica que marcar_repeticiones_sospechosas tolere líneas con JSON no-objeto

    (números, cadenas, listas, booleanos, null) y JSON sintácticamente corrupto
    sin lanzar AttributeError ni JSONDecodeError, procesando las trazas válidas.
    """
    archivo_trazas = tmp_path / "trazas_mixtas.jsonl"
    contenido = (
        "12345\n"
        "\"una cadena de texto suelta\"\n"
        "null\n"
        "[1, 2, 3]\n"
        "true\n"
        "{json_invalido_sin_comillas: 99\n"
        "\n"
        "   \n"
        '{"session_id": "sesion_resiliente", "herramienta": "consultar_catalogo_insumos"}\n'
        '{"session_id": "sesion_resiliente", "herramienta": "consultar_catalogo_insumos"}\n'
        "9999\n"
    )
    with open(archivo_trazas, "w", encoding="utf-8") as f:
        f.write(contenido)

    alertas = marcar_repeticiones_sospechosas(str(archivo_trazas))
    assert len(alertas) == 1
    assert alertas[0]["session_id"] == "sesion_resiliente"
    assert alertas[0]["herramienta"] == "consultar_catalogo_insumos"


# ============================================================================
# 12. PRUEBA DE ARGUMENTOS POSICIONALES Y CREACIÓN DE DIRECTORIOS EN REGISTRAR_TRAZA
# ============================================================================

def test_registrar_traza_argumentos_posicionales_y_creacion_directorio(tmp_path):
    """Verifica que registrar_traza soporte llamadas con argumentos posicionales

    (session_id, herramienta, entrada, observacion) sin permutar campos
    y cree automáticamente directorios intermedios inexistentes.
    """
    ruta_anidada = str(tmp_path / "subcarpeta" / "logs" / "trazas_posicional.jsonl")

    evento = registrar_traza(
        "sesion_posicional_01",
        "consultar_catalogo_insumos",
        {"nombre_insumo": "arroz"},
        "Stock 100 kg disponible",
        ruta_archivo=ruta_anidada,
    )

    assert evento["session_id"] == "sesion_posicional_01"
    assert evento["herramienta"] == "consultar_catalogo_insumos"
    assert evento["entrada"] == {"nombre_insumo": "arroz"}
    assert evento["observacion"] == "Stock 100 kg disponible"

    assert os.path.exists(ruta_anidada)
    with open(ruta_anidada, "r", encoding="utf-8") as f:
        linea = f.readline()
        datos = json.loads(linea)
        assert datos["session_id"] == "sesion_posicional_01"
        assert datos["herramienta"] == "consultar_catalogo_insumos"


# ============================================================================
# 13. PRUEBA DE CREACIÓN DE DIRECTORIOS EN REGISTRAR_APROBACION
# ============================================================================

def test_registrar_aprobacion_creacion_directorio(tmp_path):
    """Verifica que registrar_aprobacion cree directorios intermedios de forma automática."""
    ruta_anidada = str(tmp_path / "auditoria" / "sub" / "aprobaciones.jsonl")

    registro = registrar_aprobacion(
        accion="emitir_orden_compra_formal",
        aprobada=True,
        usuario="auditor_prueba",
        detalles={"orden": "OC-TEST"},
        ruta_archivo=ruta_anidada,
    )

    assert registro["accion"] == "emitir_orden_compra_formal"
    assert registro["aprobada"] is True
    assert os.path.exists(ruta_anidada)


# ============================================================================
# 14. PRUEBAS DE SEGURIDAD ANTE EOF Y EXCEPCIONES EN ESPERAR_APROBACION
# ============================================================================

def test_esperar_aprobacion_eof_y_excepcion_segura():
    """Verifica que esperar_aprobacion retorne False de forma segura ante EOF

    o ante excepciones imprevistas en streams de entrada/salida.
    """
    # 1. EOF: sys.stdin devuelve cadena vacía
    with patch("select.select") as mock_select:
        mock_select.return_value = ([sys.stdin], [], [])
        with patch.object(sys.stdin, "readline", return_value=""):
            assert esperar_aprobacion("Prueba EOF") is False

    # 2. Excepción en select.select
    with patch("select.select", side_effect=OSError("Descriptor inválido")):
        assert esperar_aprobacion("Prueba OSError") is False

    # 3. Excepción en stdout (BrokenPipeError)
    with patch("sys.stdout.write", side_effect=BrokenPipeError("Tubería rota")):
        assert esperar_aprobacion("Prueba BrokenPipe") is False


# ============================================================================
# 15. PRUEBAS DE EXCEPCIONES EN HERRAMIENTAS DE ALTO RIESGO Y ESTADOS DE BORRADOR
# ============================================================================

def test_herramientas_alto_riesgo_capturan_excepciones_y_cancelar_borrador_estados(tmp_path):
    """Verifica que las herramientas de alto riesgo capturen excepciones internas

    devolviendo str explicativo, y que cancelar_borrador_pedido maneje estados
    CANCELADO y FORMALIZADO apropiadamente.
    """
    ruta_trazas = str(tmp_path / "trazas_excepciones_alto.jsonl")
    ruta_aprobaciones = str(tmp_path / "aprobaciones_excepciones_alto.jsonl")

    with (
        patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas),
        patch("agente_con_herramientas.ARCHIVO_APROBACIONES_DEFECTO", ruta_aprobaciones),
    ):
        # 1. Simulación de fallo en emitir_orden_compra_formal
        with patch("agente_con_herramientas.esperar_aprobacion", side_effect=RuntimeError("Fallo crítico")):
            obs_oc_error = emitir_orden_compra_formal.invoke({
                "id_borrador": "BORR-TEST",
                "proveedor": "Distribuidor X",
                "monto_total_pesos": 10000.0,
            })
            assert isinstance(obs_oc_error, str)
            assert "Error interno al emitir la orden de compra formal" in obs_oc_error

        # 2. Simulación de fallo en procesar_pago_proveedor
        with patch("agente_con_herramientas.esperar_aprobacion", side_effect=RuntimeError("Fallo en conector")):
            obs_pago_error = procesar_pago_proveedor.invoke({
                "proveedor": "Distribuidor X",
                "monto_pesos": 10000.0,
                "numero_factura": "FAC-TEST",
            })
            assert isinstance(obs_pago_error, str)
            assert "Error interno al procesar el pago al proveedor" in obs_pago_error

        # 3. Flujo de estados en cancelar_borrador_pedido
        resp_crear = crear_borrador_pedido.invoke({
            "proveedor": "Distribuidor X",
            "nombre_insumo": "arroz",
            "cantidad_unidades": 10.0,
        })
        coincidencia = re.search(r"'(BORR-\d{4})'", resp_crear)
        id_nuevo = coincidencia.group(1) if coincidencia else "BORR-9999"

        # Cancelación exitosa
        obs_canc_1 = cancelar_borrador_pedido.invoke({"id_borrador": id_nuevo})
        assert "cancelado exitosamente" in obs_canc_1

        # Intento de re-cancelación (ya cancelado)
        obs_canc_2 = cancelar_borrador_pedido.invoke({"id_borrador": id_nuevo})
        assert "ya se encontraba previamente cancelado" in obs_canc_2

        # Intento de cancelación de un borrador ya formalizado
        resp_formalizar = crear_borrador_pedido.invoke({
            "proveedor": "Distribuidor Y",
            "nombre_insumo": "harina",
            "cantidad_unidades": 20.0,
        })
        coincidencia_form = re.search(r"'(BORR-\d{4})'", resp_formalizar)
        id_formalizado = coincidencia_form.group(1) if coincidencia_form else "BORR-8888"

        with patch("agente_con_herramientas.esperar_aprobacion", return_value=True):
            emitir_orden_compra_formal.invoke({
                "id_borrador": id_formalizado,
                "proveedor": "Distribuidor Y",
                "monto_total_pesos": 50000.0,
            })

        obs_canc_formalizado = cancelar_borrador_pedido.invoke({"id_borrador": id_formalizado})
        assert "ya fue formalizado" in obs_canc_formalizado


# ============================================================================
# 16. PRUEBAS DE SERIALIZACIÓN ROBUSTA DE OBJETOS NO PRIMITIVOS (DATETIME, DECIMAL)
# ============================================================================

def test_registrar_traza_y_aprobacion_serializan_objetos_no_primitivos_datetime_decimal(tmp_path):
    """Verifica que registrar_traza y registrar_aprobacion serialicen correctamente
    objetos como datetime, Decimal u otras instancias no nativas en JSON sin arrojar TypeError.
    """
    ruta_trazas = str(tmp_path / "trazas_no_primitivos.jsonl")
    ruta_aprobaciones = str(tmp_path / "aprobaciones_no_primitivos.jsonl")

    evento_traza = registrar_traza(
        session_id="sesion_tipos_avanzados",
        herramienta="calcular_cotizacion",
        entrada={"monto_decimal": Decimal("15450.75"), "fecha_registro": datetime(2026, 11, 12, 10, 30)},
        observacion={"estado": "OK", "total_moneda": Decimal("18386.39")},
        ruta_archivo=ruta_trazas,
    )
    assert evento_traza["session_id"] == "sesion_tipos_avanzados"

    evento_aprobacion = registrar_aprobacion(
        accion="procesar_pago_proveedor",
        aprobada=True,
        detalles={"monto_transferido": Decimal("500000.00"), "momento": datetime(2026, 11, 12, 10, 35)},
        ruta_archivo=ruta_aprobaciones,
    )
    assert evento_aprobacion["aprobada"] is True

    # Verificar que las líneas persistidas sean JSON válidos
    with open(ruta_trazas, "r", encoding="utf-8") as f:
        linea = f.readline().strip()
        recuperado = json.loads(linea)
        assert "15450.75" in str(recuperado["entrada"]["monto_decimal"])

    with open(ruta_aprobaciones, "r", encoding="utf-8") as f:
        linea = f.readline().strip()
        recuperado = json.loads(linea)
        assert "500000.00" in str(recuperado["detalles"]["monto_transferido"])


# ============================================================================
# 17. PRUEBAS DE VALIDACIÓN DE CANTIDADES Y MONTOS (NAN, INF, NEGATIVOS)
# ============================================================================

def test_herramientas_validan_cantidades_y_montos_nan_inf_o_negativos(tmp_path):
    """Verifica que calcular_cotizacion, crear_borrador_pedido, emitir_orden_compra_formal
    y procesar_pago_proveedor rechacen valores NaN, Inf y montos/cantidades <= 0 con mensajes explicativos.
    """
    ruta_trazas = str(tmp_path / "trazas_nan_inf.jsonl")
    ruta_aprobaciones = str(tmp_path / "aprobaciones_nan_inf.jsonl")

    with (
        patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas),
        patch("agente_con_herramientas.ARCHIVO_APROBACIONES_DEFECTO", ruta_aprobaciones),
    ):
        # 1. NaN en calcular_cotizacion
        obs_cotiz_nan = calcular_cotizacion.invoke({"nombre_insumo": "harina", "cantidad_unidades": float("nan")})
        assert "Error de validación" in obs_cotiz_nan

        # 2. Inf en crear_borrador_pedido
        obs_borr_inf = crear_borrador_pedido.invoke({
            "proveedor": "Molino Central",
            "nombre_insumo": "harina",
            "cantidad_unidades": float("inf"),
        })
        assert "Error de validación" in obs_borr_inf

        # 3. Monto <= 0 en emitir_orden_compra_formal (incluso con aprobación humana concedida)
        with patch("agente_con_herramientas.esperar_aprobacion", return_value=True):
            obs_oc_invalida = emitir_orden_compra_formal.invoke({
                "id_borrador": "BORR-9999",
                "proveedor": "Molino Central",
                "monto_total_pesos": -100.0,
            })
            assert "Error de validación" in obs_oc_invalida
            assert "estrictamente mayor a 0" in obs_oc_invalida

        # 4. Monto NaN en procesar_pago_proveedor (incluso con aprobación humana concedida)
        with patch("agente_con_herramientas.esperar_aprobacion", return_value=True):
            obs_pago_nan = procesar_pago_proveedor.invoke({
                "proveedor": "Molino Central",
                "monto_pesos": float("nan"),
                "numero_factura": "FAC-TEST-NAN",
            })
            assert "Error de validación" in obs_pago_nan
            assert "estrictamente mayor a 0" in obs_pago_nan


# ============================================================================
# 18. PRUEBA DE MANEJO SEGURO DE SESSION_ID NULO O VACÍO
# ============================================================================

def test_obtener_historial_y_ejecutar_agente_manejan_session_id_none_de_forma_segura():
    """Verifica que obtener_historial_sesion y ejecutar_agente manejen session_id=None
    sin almacenar claves None en almacen_historiales y utilizando 'sesion_predeterminada'.
    """
    historial = obtener_historial_sesion(None)
    assert historial is not None
    assert None not in sys.modules["agente_con_herramientas"].almacen_historiales

    modelo_mock = ModeloChatMock(respuestas=[AIMessage(content="Respuesta sin session_id explícito.")])
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente)
    runnable = construir_agente_con_historial(ejecutor=ejecutor)

    resultado = ejecutar_agente("Consulta prueba", session_id=None, agente_ejecutable=runnable)
    assert "Respuesta sin session_id explícito." in resultado["output"]


# ============================================================================
# 19. PRUEBA DE ESTRUCTURA EXACTA DEL DOCUMENTO DE LÍMITES DE AUTONOMÍA (R4)
# ============================================================================

def test_documento_limites_autonomia_estructura_exacta_4_secciones():
    """Verifica que docs/limites_autonomia_agente.md contenga exactamente las 4 secciones
    de segundo nivel (##) requeridas por la Unidad 2.4 y describa los 3 patrones y guía fuera de sandbox.
    """
    ruta_doc = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "docs", "limites_autonomia_agente.md"
    )
    assert os.path.exists(ruta_doc), f"No existe el documento en {ruta_doc}"

    with open(ruta_doc, "r", encoding="utf-8") as f:
        contenido = f.read()

    # Extraer encabezados de nivel 2 (##)
    encabezados_h2 = re.findall(r"^##\s+(.*)", contenido, re.MULTILINE)
    assert len(encabezados_h2) == 4, f"Se esperaban exactamente 4 secciones ##, halladas: {encabezados_h2}"

    assert encabezados_h2[0] == "Herramientas y nivel de riesgo"
    assert encabezados_h2[1] == "Herramientas que requieren aprobación humana"
    assert encabezados_h2[2] == "Revisión de trazas"
    assert encabezados_h2[3] == "Qué hacer ante un comportamiento inesperado"

    # Verificar presencia de los 3 patrones de detección críticos en Sección 3
    assert "Patrón 1: Repeticiones Consecutivas" in contenido
    assert "Patrón 2: Herramientas de Alto Riesgo sin Registro de Aprobación" in contenido
    assert "Patrón 3: Respuestas Finales Contradictorias con las Observaciones" in contenido

    # Verificar presencia de los 5 pasos de contingencia en Sección 4
    assert "Paso 1 — Contención Inmediata" in contenido
    assert "Paso 2 — Aislamiento e Inspección Forense de Trazas" in contenido
    assert "Paso 3 — Verificación y Reversión Operativa" in contenido
    assert "Paso 4 — Diagnóstico y Ajuste de Salvaguardas" in contenido
    assert "Paso 5 — Validación con Pruebas Automatizadas y Reactivación" in contenido

    # Verificar documentación de comandos para ejecución fuera del sandbox
    assert "git add" in contenido
    assert "git commit" in contenido
    assert "git push origin main" in contenido
    assert "GEMINI_API_KEY" in contenido


# ============================================================================
# 20. PRUEBA DE REGISTRAR_TRAZA Y APROBACIÓN CON ENTRADA DICT Y METADATOS
# ============================================================================

def test_registrar_traza_y_aprobacion_preservan_session_id_y_kwargs_con_dict(tmp_path):
    """Verifica que registrar_traza y registrar_aprobacion cuando reciben un diccionario
    preserven o complementen session_id y parámetros adicionales (kwargs).
    """
    ruta_trazas = str(tmp_path / "trazas_dict_input.jsonl")
    ruta_aprob = str(tmp_path / "aprob_dict_input.jsonl")

    # 1. registrar_traza con dict base y session_id explícito suplementario
    evento = registrar_traza(
        {"herramienta": "consultar_catalogo_insumos", "observacion": "Stock 40 kg"},
        session_id="sesion_dict_01",
        usuario_responsable="chef_ejecutivo",
        ruta_archivo=ruta_trazas,
    )
    assert evento["session_id"] == "sesion_dict_01"
    assert evento["usuario_responsable"] == "chef_ejecutivo"
    assert evento["herramienta"] == "consultar_catalogo_insumos"

    with open(ruta_trazas, "r", encoding="utf-8") as f:
        datos = json.loads(f.readline())
        assert datos["session_id"] == "sesion_dict_01"
        assert datos["usuario_responsable"] == "chef_ejecutivo"

    # 2. registrar_aprobacion con dict base y argumentos suplementarios
    aprob = registrar_aprobacion(
        {"accion": "emitir_orden_compra_formal"},
        aprobada=True,
        usuario="auditor_jefe",
        ruta_archivo=ruta_aprob,
    )
    assert aprob["accion"] == "emitir_orden_compra_formal"
    assert aprob["aprobada"] is True
    assert aprob["usuario"] == "auditor_jefe"

    with open(ruta_aprob, "r", encoding="utf-8") as f:
        datos_aprob = json.loads(f.readline())
        assert datos_aprob["aprobada"] is True
        assert datos_aprob["usuario"] == "auditor_jefe"


# ============================================================================
# 21. PRUEBA DE RESILIENCIA ANTE BYTES NO UTF-8 EN MARCAR_REPETICIONES
# ============================================================================

def test_marcar_repeticiones_sospechosas_tolera_bytes_no_utf8(tmp_path):
    """Verifica que marcar_repeticiones_sospechosas tolere secuencias de bytes no válidas en UTF-8
    sin arrojar UnicodeDecodeError gracias al manejo defensivo de codificación.
    """
    ruta_archivo = tmp_path / "trazas_bytes_corruptos.jsonl"
    linea_valida_1 = json.dumps({"session_id": "ses_b", "herramienta": "calcular_cotizacion"}).encode("utf-8") + b"\n"
    bytes_invalidos = b"\xff\xfe\xfa\x00\x80\n"
    linea_valida_2 = json.dumps({"session_id": "ses_b", "herramienta": "calcular_cotizacion"}).encode("utf-8") + b"\n"

    with open(ruta_archivo, "wb") as f:
        f.write(linea_valida_1)
        f.write(bytes_invalidos)
        f.write(linea_valida_2)

    alertas = marcar_repeticiones_sospechosas(str(ruta_archivo))
    assert len(alertas) == 1
    assert alertas[0]["session_id"] == "ses_b"
    assert alertas[0]["herramienta"] == "calcular_cotizacion"


# ============================================================================
# 22. PRUEBA DE MEMORIA CONVERSACIONAL MULTITURNO ACUMULADA
# ============================================================================

def test_ejecutar_agente_multiples_turnos_acumula_historial_conversacional():
    """Verifica que ejecutar_agente en múltiples turnos sobre la misma sesión
    acumule secuencialmente los pares HumanMessage y AIMessage sin perder contexto.
    """
    limpiar_historial_sesiones()

    modelo_mock = ModeloChatMock(respuestas=[
        AIMessage(content="Respuesta turno 1: Hola Chef Mario."),
        AIMessage(content="Respuesta turno 2: Harina registrada."),
        AIMessage(content="Respuesta turno 3: Orden en preparación."),
    ])
    ejecutor = construir_ejecutor_agente(agente=construir_agente_horeca(modelo_llm=modelo_mock))
    runnable = construir_agente_con_historial(ejecutor=ejecutor)

    id_sesion = "sesion_multiturno_mario"

    res_t1 = ejecutar_agente("Hola, soy el Chef Mario", session_id=id_sesion, agente_ejecutable=runnable)
    assert "Chef Mario" in res_t1["output"]

    res_t2 = ejecutar_agente("Necesito 50 kg de harina", session_id=id_sesion, agente_ejecutable=runnable)
    assert "Harina registrada" in res_t2["output"]

    res_t3 = ejecutar_agente("Por favor prepara la orden", session_id=id_sesion, agente_ejecutable=runnable)
    assert "Orden en preparación" in res_t3["output"]

    historial = obtener_historial_sesion(id_sesion)
    assert len(historial.messages) == 6  # 3 HumanMessage + 3 AIMessage

    tipos_mensajes = [type(m).__name__ for m in historial.messages]
    assert tipos_mensajes == [
        "HumanMessage", "AIMessage",
        "HumanMessage", "AIMessage",
        "HumanMessage", "AIMessage",
    ]


# ============================================================================
# 23. PRUEBAS DE VALIDACIÓN DE CADENAS VACÍAS Y ESPACIOS EN BLANCO EN HERRAMIENTAS
# ============================================================================

def test_herramientas_validan_cadenas_vacias_y_espacios_en_blanco(tmp_path):
    """Verifica que calcular_cotizacion, crear_borrador_pedido, cancelar_borrador_pedido,
    emitir_orden_compra_formal y procesar_pago_proveedor rechacen cadenas vacías o compuestas
    únicamente de espacios en blanco con mensajes explicativos de validación.
    """
    ruta_trazas = str(tmp_path / "trazas_cadenas_vacias.jsonl")
    ruta_aprob = str(tmp_path / "aprob_cadenas_vacias.jsonl")

    with (
        patch("agente_con_herramientas.ARCHIVO_TRAZAS_DEFECTO", ruta_trazas),
        patch("agente_con_herramientas.ARCHIVO_APROBACIONES_DEFECTO", ruta_aprob),
    ):
        # 1. calcular_cotizacion con nombre_insumo vacío o solo espacios
        obs_cot_vacio = calcular_cotizacion.invoke({"nombre_insumo": "   ", "cantidad_unidades": 10.0})
        assert "Error de validación" in obs_cot_vacio
        assert "nombre del insumo no puede estar vacío" in obs_cot_vacio

        # 2. crear_borrador_pedido con proveedor vacío
        obs_borr_prov_vacio = crear_borrador_pedido.invoke({
            "proveedor": "   ",
            "nombre_insumo": "harina",
            "cantidad_unidades": 10.0,
        })
        assert "Error de validación" in obs_borr_prov_vacio
        assert "proveedor no puede estar vacío" in obs_borr_prov_vacio

        # 3. crear_borrador_pedido con nombre_insumo vacío
        obs_borr_ins_vacio = crear_borrador_pedido.invoke({
            "proveedor": "Molino Central",
            "nombre_insumo": "   ",
            "cantidad_unidades": 10.0,
        })
        assert "Error de validación" in obs_borr_ins_vacio
        assert "nombre del insumo no puede estar vacío" in obs_borr_ins_vacio

        # 4. cancelar_borrador_pedido con id_borrador vacío
        obs_canc_vacio = cancelar_borrador_pedido.invoke({"id_borrador": "   "})
        assert "Error de validación" in obs_canc_vacio
        assert "identificador del borrador no puede estar vacío" in obs_canc_vacio

        # 5. emitir_orden_compra_formal con proveedor o id_borrador vacío tras aprobación
        with patch("agente_con_herramientas.esperar_aprobacion", return_value=True):
            obs_oc_vacio = emitir_orden_compra_formal.invoke({
                "id_borrador": "   ",
                "proveedor": "Molino Central",
                "monto_total_pesos": 50000.0,
            })
            assert "Error de validación" in obs_oc_vacio

        # 6. procesar_pago_proveedor con proveedor o factura vacía tras aprobación
        with patch("agente_con_herramientas.esperar_aprobacion", return_value=True):
            obs_pago_vacio = procesar_pago_proveedor.invoke({
                "proveedor": "   ",
                "monto_pesos": 50000.0,
                "numero_factura": "   ",
            })
            assert "Error de validación" in obs_pago_vacio


# ============================================================================
# 24. PRUEBA DE REGISTRAR_APROBACION CON ARCHIVO_DESTINO
# ============================================================================

def test_registrar_aprobacion_soporta_parametro_archivo_destino(tmp_path):
    """Verifica que registrar_aprobacion soporte el parámetro archivo_destino vía kwargs."""
    ruta_custom = str(tmp_path / "custom_aprobaciones.jsonl")

    registro = registrar_aprobacion(
        accion="procesar_pago_proveedor",
        aprobada=True,
        archivo_destino=ruta_custom,
    )
    assert registro["aprobada"] is True
    assert os.path.exists(ruta_custom)

    with open(ruta_custom, "r", encoding="utf-8") as f:
        linea = f.readline()
        datos = json.loads(linea)
        assert datos["accion"] == "procesar_pago_proveedor"
        assert datos["aprobada"] is True


# ============================================================================
# 25. PRUEBA DE EJECUTAR_AGENTE CON ENTRADA NONE O VACÍA
# ============================================================================

def test_ejecutar_agente_con_entrada_none_o_vacia_no_genera_error_callback():
    """Verifica que ejecutar_agente gestione entrada=None o vacía convirtiéndola
    de forma segura en texto str sin gatillar ValueErrors en los callbacks internos de LangChain.
    """
    modelo_mock = ModeloChatMock(respuestas=[
        AIMessage(content="Respuesta ante entrada nula"),
        AIMessage(content="Respuesta ante entrada vacía"),
    ])
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente)
    runnable = construir_agente_con_historial(ejecutor=ejecutor)

    res_none = ejecutar_agente(None, session_id="sesion_entrada_nula", agente_ejecutable=runnable)
    assert "Respuesta ante entrada nula" in res_none["output"]

    res_vacio = ejecutar_agente("", session_id="sesion_entrada_vacia", agente_ejecutable=runnable)
    assert "Respuesta ante entrada vacía" in res_vacio["output"]


# ============================================================================
# 26. PRUEBA DE INICIALIZACIÓN DE MODELO Y CLAVE API (.env)
# ============================================================================

def test_construir_modelo_llm_usa_endpoint_ai_studio_y_recarga_env(monkeypatch):
    """Verifica que construir_modelo_llm recargue .env con override=True y mantenga vertexai=False
    tanto para claves AIza... como para claves AQ.... de Google AI Studio.
    """
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    with patch("agente_con_herramientas.load_dotenv") as mock_load:
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaDummyKeyParaPruebasLocales123456")
        modelo_studio = construir_modelo_llm()
        mock_load.assert_called_once_with(override=True)
        assert not modelo_studio.vertexai
        assert modelo_studio.model == "gemini-3.8-flash"

    with patch("agente_con_herramientas.load_dotenv"):
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.DummyAIStudioKeyParaPruebas1234567890")
        modelo_aq = construir_modelo_llm()
        assert not modelo_aq.vertexai
        assert modelo_aq.model == "gemini-3.8-flash"


# ============================================================================
# 27. PRUEBA DE RESPUESTA MULTITURNO CON BLOQUES DE CONTENIDO GEMINI
# ============================================================================

def test_multiturno_normaliza_bloques_texto_sin_message_coercion_failure():
    """Verifica que respuestas con bloques [{'type': 'text', ...}] (Gemini 3.x)
    se normalicen a str y no fallen en el segundo turno con MESSAGE_COERCION_FAILURE.
    """
    limpiar_historial_sesiones()
    modelo_mock = ModeloChatMock(respuestas=[
        AIMessage(content=[{"type": "text", "text": "Indica qué insumo buscas.", "index": 0, "extras": {"signature": "sig"}}]),
        AIMessage(content=[{"type": "text", "text": "Tenemos queso mozzarella en stock.", "index": 0}]),
    ])
    agente = construir_agente_horeca(modelo_llm=modelo_mock)
    ejecutor = construir_ejecutor_agente(agente=agente)
    runnable = construir_agente_con_historial(ejecutor=ejecutor)

    turno_1 = ejecutar_agente("catalogo de insumos", session_id="sesion_bloques", agente_ejecutable=runnable)
    assert turno_1["output"] == "Indica qué insumo buscas."

    turno_2 = ejecutar_agente("queso mozzarella", session_id="sesion_bloques", agente_ejecutable=runnable)
    assert turno_2["output"] == "Tenemos queso mozzarella en stock."






