"""
agente_con_herramientas.py - Agente inteligente para HORECA AI Service (Unidad 2.4 / Bloques 21, 22 y 23).

Implementa un agente conversacional con llamadas a herramientas estructuradas bajo tres niveles de riesgo:
- Riesgo Bajo: Solo lectura y cálculo (consultar_catalogo_insumos, calcular_cotizacion).
- Riesgo Medio: Escritura reversible (crear_borrador_pedido, cancelar_borrador_pedido).
- Riesgo Alto: Escritura irreversible con impacto económico (emitir_orden_compra_formal, procesar_pago_proveedor).

Incorpora:
- Memoria conversacional con RunnableWithMessageHistory e InMemoryChatMessageHistory aislada por session_id.
- Control de autonomía humana (HITL) mediante select.select con tiempo límite para operaciones de alto riesgo.
- Auditoría de trazas en formato UTF-8 JSON Lines (trazas_agente.jsonl y registro_aprobaciones.jsonl).
- Detección de patrones anómalos como repeticiones consecutivas sospechosas.
"""

import json
import math
import os
import select
import sys
import warnings
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================================
# 0. CONFIGURACIÓN INICIAL Y VARIABLES DE CONTEXTO
# ============================================================================

load_dotenv(override=True)

ARCHIVO_TRAZAS_DEFECTO: str = "trazas_agente.jsonl"
ARCHIVO_APROBACIONES_DEFECTO: str = "registro_aprobaciones.jsonl"

# Variable de contexto para preservar el session_id actual durante la ejecución de herramientas
sesion_actual: ContextVar[str] = ContextVar("sesion_actual", default="sesion_predeterminada")

# Almacén de historiales conversacionales aislados por identificador de sesión
almacen_historiales: Dict[str, InMemoryChatMessageHistory] = {}

# Catálogo mock de insumos gastronómicos para conexión entre negocios HORECA y proveedores
CATALOGO_INSUMOS_HORECA: Dict[str, Dict[str, Any]] = {
    "harina": {
        "nombre": "Harina de trigo fuerza 000",
        "stock": 150.0,
        "unidad": "kg",
        "proveedor": "Molino Central",
        "precio_unitario_pesos": 850.0,
    },
    "aceite de oliva": {
        "nombre": "Aceite de oliva extra virgen",
        "stock": 60.0,
        "unidad": "litros",
        "proveedor": "Distribuidora Valle Central",
        "precio_unitario_pesos": 6200.0,
    },
    "tomate triturado": {
        "nombre": "Tomate triturado en lata 2.5 kg",
        "stock": 80.0,
        "unidad": "latas",
        "proveedor": "Agrícola El Sol",
        "precio_unitario_pesos": 3400.0,
    },
    "queso mozzarella": {
        "nombre": "Queso mozzarella barra gastronómica",
        "stock": 45.0,
        "unidad": "kg",
        "proveedor": "Lácteos del Sur",
        "precio_unitario_pesos": 5900.0,
    },
    "carne vacuno lomo": {
        "nombre": "Lomo liso vacuno categoría V",
        "stock": 30.0,
        "unidad": "kg",
        "proveedor": "Frigorífico Andino",
        "precio_unitario_pesos": 11500.0,
    },
    "pechuga de pollo": {
        "nombre": "Pechuga de pollo deshuesada fresca",
        "stock": 75.0,
        "unidad": "kg",
        "proveedor": "Avícola Campestre",
        "precio_unitario_pesos": 4200.0,
    },
    "arroz": {
        "nombre": "Arroz grano largo grado 1",
        "stock": 100.0,
        "unidad": "kg",
        "proveedor": "Arrocera Nacional",
        "precio_unitario_pesos": 1250.0,
    },
}

# Registros transaccionales en memoria para simular la plataforma B2B HORECA
BORRADORES_PEDIDOS: Dict[str, Dict[str, Any]] = {}
ORDENES_COMPRA_EMITIDAS: Dict[str, Dict[str, Any]] = {}
PAGOS_PROCESADOS: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# 1. AUDITORÍA DE TRAZAS Y CONTROL DE AUTONOMÍA HUMANA
# ============================================================================

def esperar_aprobacion(mensaje: str, tiempo_limite_segundos: int = 300) -> bool:
    """Solicita autorización humana explícita mediante sys.stdin con temporizador select.select.

    Retorna True si el operador autoriza expresamente ('s', 'si', 'sí', 'y', 'yes').
    Retorna False de forma segura ante rechazo, expiración del tiempo límite o error de lectura.
    """
    try:
        sys.stdout.write(f"\n[CONTROL DE AUTONOMÍA HUMANA] {mensaje}\n")
        sys.stdout.write(
            f"¿Autoriza ejecutar esta acción de ALTO RIESGO? (s/n) [Tiempo límite: {tiempo_limite_segundos}s]: "
        )
        sys.stdout.flush()

        lectores_disponibles, _, _ = select.select([sys.stdin], [], [], tiempo_limite_segundos)
        if lectores_disponibles:
            linea_ingresada = sys.stdin.readline()
            if not linea_ingresada:
                # Llegada a fin de archivo (EOF)
                return False
            respuesta = linea_ingresada.strip().lower()
            aprobado = respuesta in ("s", "si", "sí", "y", "yes")
            if not aprobado:
                sys.stdout.write("[ACCESO DENEGADO] Operación cancelada por el operador humano.\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("[AUTORIZADO] Operación aprobada por el operador humano.\n")
                sys.stdout.flush()
            return aprobado
        else:
            sys.stdout.write(
                f"\n[TIEMPO EXPIRADO] Se superó el límite de {tiempo_limite_segundos}s sin respuesta. Operación cancelada.\n"
            )
            sys.stdout.flush()
            return False
    except Exception as error_lectura:
        try:
            sys.stdout.write(
                f"\n[ERROR EN APROBACIÓN] No fue posible capturar la confirmación: {error_lectura}. Operación cancelada.\n"
            )
            sys.stdout.flush()
        except Exception:
            pass
        return False


def registrar_aprobacion(
    accion: Optional[Any] = None,
    aprobada: Optional[bool] = None,
    usuario: str = "operador_supervisor",
    detalles: Optional[Dict[str, Any]] = None,
    ruta_archivo: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Almacena el registro de autorización o rechazo humano en formato UTF-8 JSON Lines."""
    archivo_destino = ruta_archivo or kwargs.pop("archivo_destino", None) or ARCHIVO_APROBACIONES_DEFECTO
    directorio = os.path.dirname(archivo_destino)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    if isinstance(accion, dict):
        registro = dict(accion)
        if aprobada is not None and "aprobada" not in registro:
            registro["aprobada"] = aprobada
        if usuario and "usuario" not in registro:
            registro["usuario"] = usuario
        if detalles and "detalles" not in registro:
            registro["detalles"] = detalles
        for clave, valor in kwargs.items():
            if clave not in registro:
                registro[clave] = valor
    else:
        registro = {
            "timestamp": datetime.now().isoformat(),
            "accion": accion,
            "aprobada": aprobada,
            "usuario": usuario,
            "detalles": detalles or {},
            **kwargs,
        }

    if "timestamp" not in registro and "marca_tiempo" not in registro:
        registro["timestamp"] = datetime.now().isoformat()

    linea_json = json.dumps(registro, ensure_ascii=False, default=str)
    with open(archivo_destino, "a", encoding="utf-8") as archivo_aprobaciones:
        archivo_aprobaciones.write(linea_json + "\n")
    return registro


def registrar_traza(
    evento_o_sesion: Optional[Any] = None,
    herramienta: Optional[str] = None,
    entrada: Optional[Any] = None,
    observacion: Optional[Any] = None,
    session_id: Optional[str] = None,
    ruta_archivo: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Vuelca cada acción y observación en formato UTF-8 JSON Lines al archivo de trazas."""
    archivo_destino = ruta_archivo or kwargs.pop("archivo_destino", None) or ARCHIVO_TRAZAS_DEFECTO
    directorio = os.path.dirname(archivo_destino)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    if isinstance(evento_o_sesion, dict):
        evento = dict(evento_o_sesion)
        if session_id and "session_id" not in evento:
            evento["session_id"] = str(session_id)
        elif "session_id" not in evento:
            evento["session_id"] = sesion_actual.get()
        if herramienta and "herramienta" not in evento:
            evento["herramienta"] = herramienta
        if entrada is not None and "entrada" not in evento:
            evento["entrada"] = entrada
        if observacion is not None and "observacion" not in evento:
            evento["observacion"] = observacion
        for clave, valor in kwargs.items():
            if clave not in evento:
                evento[clave] = valor
    else:
        id_ses = (
            session_id
            or (str(evento_o_sesion) if evento_o_sesion is not None else None)
            or sesion_actual.get()
        )
        evento = {
            "session_id": str(id_ses),
            "herramienta": herramienta or kwargs.pop("accion", None) or "desconocida",
            "entrada": entrada,
            "observacion": observacion,
            **kwargs,
        }

    if "timestamp" not in evento and "marca_tiempo" not in evento:
        evento["timestamp"] = datetime.now().isoformat()

    # Preservar sinónimos para máxima interoperabilidad con auditorías
    if "observacion" not in evento and "salida" in evento:
        evento["observacion"] = evento["salida"]
    elif "salida" not in evento and "observacion" in evento:
        evento["salida"] = evento["observacion"]

    if "herramienta" not in evento and "accion" in evento:
        evento["herramienta"] = evento["accion"]
    elif "accion" not in evento and "herramienta" in evento:
        evento["accion"] = evento["herramienta"]

    linea_json = json.dumps(evento, ensure_ascii=False, default=str)
    with open(archivo_destino, "a", encoding="utf-8") as archivo_trazas:
        archivo_trazas.write(linea_json + "\n")
    return evento


def marcar_repeticiones_sospechosas(ruta_archivo: str = ARCHIVO_TRAZAS_DEFECTO) -> List[Dict[str, Any]]:
    """Lee trazas_agente.jsonl en UTF-8, manejando FileNotFoundError limpiamente,

    y alerta cuando en una sesión la misma herramienta sea invocada de forma consecutiva
    (herramientas[i] == herramientas[i - 1]).
    """
    alertas: List[Dict[str, Any]] = []
    try:
        with open(ruta_archivo, "r", encoding="utf-8", errors="replace") as archivo:
            lineas = archivo.readlines()
    except FileNotFoundError:
        return []

    # Agrupar las herramientas invocadas por cada sesión manteniendo el orden temporal
    eventos_por_sesion: Dict[str, List[Dict[str, Any]]] = {}
    for indice_linea, linea in enumerate(lineas, start=1):
        linea_limpia = linea.strip()
        if not linea_limpia:
            continue
        try:
            registro = json.loads(linea_limpia)
        except json.JSONDecodeError:
            continue

        if not isinstance(registro, dict):
            continue

        nombre_herramienta = registro.get("herramienta") or registro.get("accion")
        if not nombre_herramienta:
            continue

        id_sesion = str(registro.get("session_id") or "sesion_predeterminada")
        if id_sesion not in eventos_por_sesion:
            eventos_por_sesion[id_sesion] = []

        registro["_linea_archivo"] = indice_linea
        eventos_por_sesion[id_sesion].append(registro)

    for id_sesion, lista_eventos in eventos_por_sesion.items():
        for i in range(1, len(lista_eventos)):
            herramienta_actual = lista_eventos[i].get("herramienta") or lista_eventos[i].get("accion")
            herramienta_anterior = lista_eventos[i - 1].get("herramienta") or lista_eventos[i - 1].get("accion")

            if herramienta_actual and herramienta_actual == herramienta_anterior:
                alertas.append({
                    "session_id": id_sesion,
                    "herramienta": herramienta_actual,
                    "indice_repeticion": i,
                    "linea_previa": lista_eventos[i - 1].get("_linea_archivo"),
                    "linea_actual": lista_eventos[i].get("_linea_archivo"),
                    "mensaje": (
                        f"Alerta: Invocación consecutiva sospechosa de la herramienta '{herramienta_actual}' "
                        f"en la sesión '{id_sesion}' (líneas {lista_eventos[i - 1].get('_linea_archivo')} "
                        f"y {lista_eventos[i].get('_linea_archivo')})."
                    ),
                    "evento_previo": lista_eventos[i - 1],
                    "evento_actual": lista_eventos[i],
                })

    return alertas


# ============================================================================
# 2. ESQUEMAS PYDANTIC (ARGS_SCHEMA PARA HERRAMIENTAS CON > 1 PARÁMETRO)
# ============================================================================

class EsquemaCalcularCotizacion(BaseModel):
    """Esquema para calcular la cotización estimada de un insumo gastronómico."""

    nombre_insumo: str = Field(
        description="Nombre o término de búsqueda del insumo gastronómico a cotizar (ej: 'harina', 'tomate triturado')"
    )
    cantidad_unidades: float = Field(
        description="Cantidad requerida en unidades o kilogramos del insumo para calcular el costo total"
    )


class EsquemaCrearBorradorPedido(BaseModel):
    """Esquema para la creación preliminar reversible de un borrador de pedido."""

    proveedor: str = Field(
        description="Nombre o razón social del proveedor gastronómico seleccionado"
    )
    nombre_insumo: str = Field(
        description="Nombre del insumo o producto a incorporar en el borrador de pedido"
    )
    cantidad_unidades: float = Field(
        description="Cantidad de unidades o kilogramos solicitados en la orden borrador"
    )


class EsquemaEmitirOrdenCompra(BaseModel):
    """Esquema para la emisión formal e irreversible de una orden de compra vinculante."""

    id_borrador: str = Field(
        description="Identificador del borrador de pedido previo a formalizar (ej: 'BORR-0001')"
    )
    proveedor: str = Field(
        description="Nombre formal del proveedor gastronómico destinatario de la orden"
    )
    monto_total_pesos: float = Field(
        description="Monto financiero total en pesos chilenos comprometido en la orden de compra vinculante"
    )


class EsquemaProcesarPago(BaseModel):
    """Esquema para la ejecución de una transferencia monetaria irreversible a un proveedor."""

    proveedor: str = Field(
        description="Nombre o razón social del proveedor gastronómico receptor de la transferencia"
    )
    monto_pesos: float = Field(
        description="Monto monetario exacto en pesos chilenos a transferir desde el negocio comprador al proveedor"
    )
    numero_factura: str = Field(
        description="Número de folio o factura comercial asociada a la obligación de pago"
    )


# ============================================================================
# 3. HERRAMIENTAS HORECA EN TRES NIVELES DE RIESGO
# ============================================================================

# ----------------------------------------------------------------------------
# NIVEL 1: RIESGO BAJO (Solo lectura y cálculos determinísticos)
# ----------------------------------------------------------------------------

@tool("consultar_catalogo_insumos")
def consultar_catalogo_insumos(nombre_insumo: str) -> str:
    """Consulta disponibilidad de stock, unidad de medida, proveedor habitual y precio de un insumo."""
    id_sesion = sesion_actual.get()
    try:
        if not isinstance(nombre_insumo, str) or not nombre_insumo.strip():
            observacion = "Error: El nombre del insumo no puede estar vacío."
            registrar_traza(
                session_id=id_sesion,
                herramienta="consultar_catalogo_insumos",
                nivel_riesgo="bajo",
                entrada={"nombre_insumo": nombre_insumo},
                observacion=observacion,
            )
            return observacion

        termino = nombre_insumo.strip().lower()
        encontrados = []
        for clave, datos in CATALOGO_INSUMOS_HORECA.items():
            if termino in clave or termino in datos["nombre"].lower():
                encontrados.append(datos)

        if encontrados:
            lineas_resultado = [f"Se encontraron {len(encontrados)} insumo(s) en el catálogo:"]
            for item in encontrados:
                lineas_resultado.append(
                    f"- {item['nombre']}: Stock {item['stock']} {item['unidad']} | "
                    f"Proveedor: {item['proveedor']} | "
                    f"Precio unitario: ${item['precio_unitario_pesos']:,.2f} CLP"
                )
            observacion = "\n".join(lineas_resultado)
        else:
            disponibles = ", ".join(list(CATALOGO_INSUMOS_HORECA.keys())[:5])
            observacion = (
                f"No se encontró el insumo '{nombre_insumo}' en el catálogo gastronómico. "
                f"Insumos de referencia disponibles: {disponibles}."
            )

        registrar_traza(
            session_id=id_sesion,
            herramienta="consultar_catalogo_insumos",
            nivel_riesgo="bajo",
            entrada={"nombre_insumo": nombre_insumo},
            observacion=observacion,
        )
        return observacion

    except Exception as error_ejecucion:
        observacion_error = f"Error interno al consultar el catálogo de insumos: {str(error_ejecucion)}"
        registrar_traza(
            session_id=id_sesion,
            herramienta="consultar_catalogo_insumos",
            nivel_riesgo="bajo",
            entrada={"nombre_insumo": nombre_insumo},
            observacion=observacion_error,
        )
        return observacion_error


@tool("calcular_cotizacion", args_schema=EsquemaCalcularCotizacion)
def calcular_cotizacion(nombre_insumo: str, cantidad_unidades: float) -> str:
    """Calcula el subtotal estimado, IVA (19%) y total para una compra gastronómica sin efectos colaterales."""
    id_sesion = sesion_actual.get()
    try:
        if not isinstance(nombre_insumo, str) or not nombre_insumo.strip():
            observacion = "Error de validación: El nombre del insumo no puede estar vacío."
            registrar_traza(
                session_id=id_sesion,
                herramienta="calcular_cotizacion",
                nivel_riesgo="bajo",
                entrada={"nombre_insumo": nombre_insumo, "cantidad_unidades": cantidad_unidades},
                observacion=observacion,
            )
            return observacion

        if (
            not isinstance(cantidad_unidades, (int, float))
            or math.isnan(cantidad_unidades)
            or math.isinf(cantidad_unidades)
            or cantidad_unidades <= 0
        ):
            observacion = "Error de validación: La cantidad de unidades solicitada debe ser estrictamente mayor a 0."
            registrar_traza(
                session_id=id_sesion,
                herramienta="calcular_cotizacion",
                nivel_riesgo="bajo",
                entrada={"nombre_insumo": nombre_insumo, "cantidad_unidades": cantidad_unidades},
                observacion=observacion,
            )
            return observacion

        termino = nombre_insumo.strip().lower()
        precio_unitario = None
        nombre_oficial = nombre_insumo
        unidad = "unidades"

        for clave, datos in CATALOGO_INSUMOS_HORECA.items():
            if termino in clave or termino in datos["nombre"].lower():
                precio_unitario = datos["precio_unitario_pesos"]
                nombre_oficial = datos["nombre"]
                unidad = datos["unidad"]
                break

        if precio_unitario is None:
            # Precio promedio referencial para insumos generales no listados
            precio_unitario = 3500.0
            nombre_oficial = f"{nombre_insumo} (precio referencial de mercado)"

        subtotal = round(cantidad_unidades * precio_unitario, 2)
        iva = round(subtotal * 0.19, 2)
        total = round(subtotal + iva, 2)

        observacion = (
            f"Cotización estimada para '{nombre_oficial}':\n"
            f"- Cantidad: {cantidad_unidades} {unidad}\n"
            f"- Precio unitario: ${precio_unitario:,.2f} CLP\n"
            f"- Subtotal neto: ${subtotal:,.2f} CLP\n"
            f"- IVA (19%): ${iva:,.2f} CLP\n"
            f"- Total estimado con IVA: ${total:,.2f} CLP"
        )

        registrar_traza(
            session_id=id_sesion,
            herramienta="calcular_cotizacion",
            nivel_riesgo="bajo",
            entrada={"nombre_insumo": nombre_insumo, "cantidad_unidades": cantidad_unidades},
            observacion=observacion,
        )
        return observacion

    except Exception as error_ejecucion:
        observacion_error = f"Error interno al calcular cotización: {str(error_ejecucion)}"
        registrar_traza(
            session_id=id_sesion,
            herramienta="calcular_cotizacion",
            nivel_riesgo="bajo",
            entrada={"nombre_insumo": nombre_insumo, "cantidad_unidades": cantidad_unidades},
            observacion=observacion_error,
        )
        return observacion_error


# ----------------------------------------------------------------------------
# NIVEL 2: RIESGO MEDIO (Escritura reversible en estado preliminar)
# ----------------------------------------------------------------------------

@tool("crear_borrador_pedido", args_schema=EsquemaCrearBorradorPedido)
def crear_borrador_pedido(proveedor: str, nombre_insumo: str, cantidad_unidades: float) -> str:
    """Crea una orden preliminar en estado 'BORRADOR'. Es una operación reversible que no compromete fondos."""
    id_sesion = sesion_actual.get()
    try:
        if not isinstance(proveedor, str) or not proveedor.strip():
            observacion = "Error de validación: El proveedor no puede estar vacío."
            registrar_traza(
                session_id=id_sesion,
                herramienta="crear_borrador_pedido",
                nivel_riesgo="medio",
                entrada={
                    "proveedor": proveedor,
                    "nombre_insumo": nombre_insumo,
                    "cantidad_unidades": cantidad_unidades,
                },
                observacion=observacion,
            )
            return observacion

        if not isinstance(nombre_insumo, str) or not nombre_insumo.strip():
            observacion = "Error de validación: El nombre del insumo no puede estar vacío."
            registrar_traza(
                session_id=id_sesion,
                herramienta="crear_borrador_pedido",
                nivel_riesgo="medio",
                entrada={
                    "proveedor": proveedor,
                    "nombre_insumo": nombre_insumo,
                    "cantidad_unidades": cantidad_unidades,
                },
                observacion=observacion,
            )
            return observacion

        if (
            not isinstance(cantidad_unidades, (int, float))
            or math.isnan(cantidad_unidades)
            or math.isinf(cantidad_unidades)
            or cantidad_unidades <= 0
        ):
            observacion = "Error de validación: La cantidad para el borrador debe ser superior a 0."
            registrar_traza(
                session_id=id_sesion,
                herramienta="crear_borrador_pedido",
                nivel_riesgo="medio",
                entrada={
                    "proveedor": proveedor,
                    "nombre_insumo": nombre_insumo,
                    "cantidad_unidades": cantidad_unidades,
                },
                observacion=observacion,
            )
            return observacion

        numero_borrador = len(BORRADORES_PEDIDOS) + 1
        id_borrador = f"BORR-{numero_borrador:04d}"

        registro_borrador = {
            "id_borrador": id_borrador,
            "session_id": id_sesion,
            "proveedor": proveedor.strip(),
            "nombre_insumo": nombre_insumo.strip(),
            "cantidad_unidades": cantidad_unidades,
            "estado": "BORRADOR",
            "creado_en": datetime.now().isoformat(),
        }
        BORRADORES_PEDIDOS[id_borrador] = registro_borrador

        observacion = (
            f"Borrador de pedido creado exitosamente con identificador '{id_borrador}'.\n"
            f"- Proveedor: {proveedor}\n"
            f"- Insumo: {nombre_insumo}\n"
            f"- Cantidad: {cantidad_unidades}\n"
            f"- Estado: BORRADOR (Operación reversible. Puede modificarse o cancelarse antes de emitir orden formal)."
        )

        registrar_traza(
            session_id=id_sesion,
            herramienta="crear_borrador_pedido",
            nivel_riesgo="medio",
            entrada={
                "proveedor": proveedor,
                "nombre_insumo": nombre_insumo,
                "cantidad_unidades": cantidad_unidades,
            },
            observacion=observacion,
        )
        return observacion

    except Exception as error_ejecucion:
        observacion_error = f"Error interno al crear el borrador de pedido: {str(error_ejecucion)}"
        registrar_traza(
            session_id=id_sesion,
            herramienta="crear_borrador_pedido",
            nivel_riesgo="medio",
            entrada={
                "proveedor": proveedor,
                "nombre_insumo": nombre_insumo,
                "cantidad_unidades": cantidad_unidades,
            },
            observacion=observacion_error,
        )
        return observacion_error


@tool("cancelar_borrador_pedido")
def cancelar_borrador_pedido(id_borrador: str) -> str:
    """Cancela o revierte un borrador de pedido previo sin consecuencias contractuales ni financieras."""
    id_sesion = sesion_actual.get()
    try:
        if not isinstance(id_borrador, str) or not id_borrador.strip():
            observacion = "Error de validación: El identificador del borrador no puede estar vacío."
            registrar_traza(
                session_id=id_sesion,
                herramienta="cancelar_borrador_pedido",
                nivel_riesgo="medio",
                entrada={"id_borrador": id_borrador},
                observacion=observacion,
            )
            return observacion

        id_limpio = id_borrador.strip().upper()
        if id_limpio not in BORRADORES_PEDIDOS:
            observacion = (
                f"No se encontró el borrador '{id_borrador}'. Verifique el identificador registrado."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="cancelar_borrador_pedido",
                nivel_riesgo="medio",
                entrada={"id_borrador": id_borrador},
                observacion=observacion,
            )
            return observacion

        borrador = BORRADORES_PEDIDOS[id_limpio]
        if borrador["estado"] == "CANCELADO":
            observacion = f"El borrador '{id_limpio}' ya se encontraba previamente cancelado."
        elif borrador["estado"] == "FORMALIZADO":
            observacion = (
                f"No se puede cancelar el borrador '{id_limpio}' porque ya fue formalizado "
                "en una Orden de Compra definitiva."
            )
        else:
            borrador["estado"] = "CANCELADO"
            borrador["cancelado_en"] = datetime.now().isoformat()
            observacion = (
                f"Borrador '{id_limpio}' cancelado exitosamente. No se generarán cobros ni despachos con el proveedor."
            )

        registrar_traza(
            session_id=id_sesion,
            herramienta="cancelar_borrador_pedido",
            nivel_riesgo="medio",
            entrada={"id_borrador": id_borrador},
            observacion=observacion,
        )
        return observacion

    except Exception as error_ejecucion:
        observacion_error = f"Error interno al cancelar el borrador de pedido: {str(error_ejecucion)}"
        registrar_traza(
            session_id=id_sesion,
            herramienta="cancelar_borrador_pedido",
            nivel_riesgo="medio",
            entrada={"id_borrador": id_borrador},
            observacion=observacion_error,
        )
        return observacion_error


# ----------------------------------------------------------------------------
# NIVEL 3: RIESGO ALTO (Escritura irreversible / Impacto financiero)
# ----------------------------------------------------------------------------

@tool("emitir_orden_compra_formal", args_schema=EsquemaEmitirOrdenCompra)
def emitir_orden_compra_formal(
    id_borrador: str, proveedor: str, monto_total_pesos: float
) -> str:
    """Emite formalmente una Orden de Compra vinculante. Requiere autorización humana obligatoria."""
    id_sesion = sesion_actual.get()
    try:
        mensaje_solicitud = (
            f"Emisión vinculante de Orden de Compra definitiva basada en '{id_borrador}' "
            f"para el proveedor '{proveedor}' por un total de ${monto_total_pesos:,.2f} CLP."
        )

        # Control de autonomía humana obligatorio: única invocación permitida
        autorizado = esperar_aprobacion(mensaje_solicitud, tiempo_limite_segundos=300)

        # Registro del evento de auditoría de autorización
        registrar_aprobacion(
            accion="emitir_orden_compra_formal",
            aprobada=autorizado,
            detalles={
                "session_id": id_sesion,
                "id_borrador": id_borrador,
                "proveedor": proveedor,
                "monto_total_pesos": monto_total_pesos,
            },
        )

        if not autorizado:
            observacion = (
                f"OPERACIÓN CANCELADA: La emisión formal de la orden de compra para '{proveedor}' "
                "fue rechazada por el operador humano o expiró el tiempo de espera (300s)."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="emitir_orden_compra_formal",
                nivel_riesgo="alto",
                entrada={
                    "id_borrador": id_borrador,
                    "proveedor": proveedor,
                    "monto_total_pesos": monto_total_pesos,
                },
                observacion=observacion,
                aprobado_por_humano=False,
            )
            return observacion

        if (
            not isinstance(proveedor, str)
            or not proveedor.strip()
            or not isinstance(id_borrador, str)
            or not id_borrador.strip()
        ):
            observacion = (
                "Error de validación: El proveedor y el ID de borrador no pueden estar vacíos."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="emitir_orden_compra_formal",
                nivel_riesgo="alto",
                entrada={
                    "id_borrador": id_borrador,
                    "proveedor": proveedor,
                    "monto_total_pesos": monto_total_pesos,
                },
                observacion=observacion,
                aprobado_por_humano=True,
            )
            return observacion

        if (
            not isinstance(monto_total_pesos, (int, float))
            or math.isnan(monto_total_pesos)
            or math.isinf(monto_total_pesos)
            or monto_total_pesos <= 0
        ):
            observacion = (
                "Error de validación: El monto total de la orden de compra debe ser estrictamente mayor a 0."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="emitir_orden_compra_formal",
                nivel_riesgo="alto",
                entrada={
                    "id_borrador": id_borrador,
                    "proveedor": proveedor,
                    "monto_total_pesos": monto_total_pesos,
                },
                observacion=observacion,
                aprobado_por_humano=True,
            )
            return observacion

        numero_orden = len(ORDENES_COMPRA_EMITIDAS) + 1
        id_orden = f"OC-2026-{numero_orden:04d}"

        registro_orden = {
            "id_orden": id_orden,
            "id_borrador": id_borrador,
            "session_id": id_sesion,
            "proveedor": proveedor,
            "monto_total_pesos": monto_total_pesos,
            "estado": "EMITIDA_VINCULANTE",
            "emitida_en": datetime.now().isoformat(),
        }
        ORDENES_COMPRA_EMITIDAS[id_orden] = registro_orden

        id_limpio = id_borrador.strip().upper()
        if id_limpio in BORRADORES_PEDIDOS:
            BORRADORES_PEDIDOS[id_limpio]["estado"] = "FORMALIZADO"
            BORRADORES_PEDIDOS[id_limpio]["id_orden"] = id_orden

        observacion = (
            f"ORDEN DE COMPRA EMITIDA CON ÉXITO: Folio '{id_orden}'.\n"
            f"- Proveedor: {proveedor}\n"
            f"- Monto total vinculado: ${monto_total_pesos:,.2f} CLP\n"
            f"- Borrador asociado: {id_borrador}\n"
            "- Autorización humana verificada y registrada en auditoría."
        )

        registrar_traza(
            session_id=id_sesion,
            herramienta="emitir_orden_compra_formal",
            nivel_riesgo="alto",
            entrada={
                "id_borrador": id_borrador,
                "proveedor": proveedor,
                "monto_total_pesos": monto_total_pesos,
            },
            observacion=observacion,
            aprobado_por_humano=True,
        )
        return observacion

    except Exception as error_ejecucion:
        observacion_error = f"Error interno al emitir la orden de compra formal: {str(error_ejecucion)}"
        registrar_traza(
            session_id=id_sesion,
            herramienta="emitir_orden_compra_formal",
            nivel_riesgo="alto",
            entrada={
                "id_borrador": id_borrador,
                "proveedor": proveedor,
                "monto_total_pesos": monto_total_pesos,
            },
            observacion=observacion_error,
        )
        return observacion_error


@tool("procesar_pago_proveedor", args_schema=EsquemaProcesarPago)
def procesar_pago_proveedor(proveedor: str, monto_pesos: float, numero_factura: str) -> str:
    """Ejecuta una transferencia bancaria definitiva a un proveedor. Requiere autorización humana obligatoria."""
    id_sesion = sesion_actual.get()
    try:
        mensaje_solicitud = (
            f"Transferencia bancaria irreversible de ${monto_pesos:,.2f} CLP al proveedor '{proveedor}' "
            f"en liquidación de la factura #{numero_factura}."
        )

        # Control de autonomía humana obligatorio: única invocación permitida
        autorizado = esperar_aprobacion(mensaje_solicitud, tiempo_limite_segundos=300)

        # Registro del evento de auditoría de autorización
        registrar_aprobacion(
            accion="procesar_pago_proveedor",
            aprobada=autorizado,
            detalles={
                "session_id": id_sesion,
                "proveedor": proveedor,
                "monto_pesos": monto_pesos,
                "numero_factura": numero_factura,
            },
        )

        if not autorizado:
            observacion = (
                f"PAGO CANCELADO: La transferencia de ${monto_pesos:,.2f} CLP al proveedor '{proveedor}' "
                "fue rechazada por el operador humano o expiró el tiempo de espera (300s)."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="procesar_pago_proveedor",
                nivel_riesgo="alto",
                entrada={
                    "proveedor": proveedor,
                    "monto_pesos": monto_pesos,
                    "numero_factura": numero_factura,
                },
                observacion=observacion,
                aprobado_por_humano=False,
            )
            return observacion

        if (
            not isinstance(proveedor, str)
            or not proveedor.strip()
            or not isinstance(numero_factura, str)
            or not numero_factura.strip()
        ):
            observacion = (
                "Error de validación: El proveedor y el número de factura no pueden estar vacíos."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="procesar_pago_proveedor",
                nivel_riesgo="alto",
                entrada={
                    "proveedor": proveedor,
                    "monto_pesos": monto_pesos,
                    "numero_factura": numero_factura,
                },
                observacion=observacion,
                aprobado_por_humano=True,
            )
            return observacion

        if (
            not isinstance(monto_pesos, (int, float))
            or math.isnan(monto_pesos)
            or math.isinf(monto_pesos)
            or monto_pesos <= 0
        ):
            observacion = (
                "Error de validación: El monto a pagar debe ser estrictamente mayor a 0."
            )
            registrar_traza(
                session_id=id_sesion,
                herramienta="procesar_pago_proveedor",
                nivel_riesgo="alto",
                entrada={
                    "proveedor": proveedor,
                    "monto_pesos": monto_pesos,
                    "numero_factura": numero_factura,
                },
                observacion=observacion,
                aprobado_por_humano=True,
            )
            return observacion

        numero_pago = len(PAGOS_PROCESADOS) + 1
        id_pago = f"PAGO-2026-{numero_pago:04d}"

        registro_pago = {
            "id_pago": id_pago,
            "session_id": id_sesion,
            "proveedor": proveedor,
            "monto_pesos": monto_pesos,
            "numero_factura": numero_factura,
            "estado": "TRANSFERIDO_EXITOSO",
            "procesado_en": datetime.now().isoformat(),
        }
        PAGOS_PROCESADOS[id_pago] = registro_pago

        observacion = (
            f"TRANSFERENCIA BANCARIA COMPLETADA EXITOSAMENTE: Comprobante '{id_pago}'.\n"
            f"- Beneficiario: {proveedor}\n"
            f"- Monto liquidado: ${monto_pesos:,.2f} CLP\n"
            f"- Factura saldada: #{numero_factura}\n"
            "- Operación irreversible autorizada por el supervisor y archivada en auditoría."
        )

        registrar_traza(
            session_id=id_sesion,
            herramienta="procesar_pago_proveedor",
            nivel_riesgo="alto",
            entrada={
                "proveedor": proveedor,
                "monto_pesos": monto_pesos,
                "numero_factura": numero_factura,
            },
            observacion=observacion,
            aprobado_por_humano=True,
        )
        return observacion

    except Exception as error_ejecucion:
        observacion_error = f"Error interno al procesar el pago al proveedor: {str(error_ejecucion)}"
        registrar_traza(
            session_id=id_sesion,
            herramienta="procesar_pago_proveedor",
            nivel_riesgo="alto",
            entrada={
                "proveedor": proveedor,
                "monto_pesos": monto_pesos,
                "numero_factura": numero_factura,
            },
            observacion=observacion_error,
        )
        return observacion_error


# Lista de herramientas agrupadas para el agente HORECA
HERRAMIENTAS_HORECA: List[Any] = [
    consultar_catalogo_insumos,
    calcular_cotizacion,
    crear_borrador_pedido,
    cancelar_borrador_pedido,
    emitir_orden_compra_formal,
    procesar_pago_proveedor,
]

NIVELES_RIESGO_HERRAMIENTAS: Dict[str, str] = {
    "consultar_catalogo_insumos": "bajo",
    "calcular_cotizacion": "bajo",
    "crear_borrador_pedido": "medio",
    "cancelar_borrador_pedido": "medio",
    "emitir_orden_compra_formal": "alto",
    "procesar_pago_proveedor": "alto",
}


# ============================================================================
# 4. PLANTILLA DE PROMPT CON ORDEN ESTRICTO DE PLACEHOLDERS (R2)
# ============================================================================

PROMPT_SISTEMA_HORECA: str = (
    "Eres el Agente Asistente Inteligente de Operaciones y Compras de HORECA AI Service, "
    "una plataforma que conecta negocios del sector HORECA (hoteles, restaurantes y cafeterías) "
    "con proveedores mayoristas para optimizar el abastecimiento, el control de costos y la gestión de pedidos.\n\n"
    "Cuentas con herramientas especializadas clasificadas en tres niveles de riesgo:\n"
    "1. RIESGO BAJO (Solo lectura y cálculos determinísticos): 'consultar_catalogo_insumos' y 'calcular_cotizacion'. "
    "Se ejecutan inmediatamente sin impacto colateral.\n"
    "2. RIESGO MEDIO (Escritura reversible): 'crear_borrador_pedido' y 'cancelar_borrador_pedido'. "
    "Permiten generar o revertir propuestas preliminares sin comprometer fondos.\n"
    "3. RIESGO ALTO (Escritura irreversible / Impacto financiero): 'emitir_orden_compra_formal' y 'procesar_pago_proveedor'. "
    "Estas operaciones requerirán autorización humana explícita antes de ejecutarse.\n\n"
    "Directrices operativas:\n"
    "- Si el usuario saluda o realiza consultas generales que no requieren datos de catálogo, responde directamente sin invocar herramientas.\n"
    "- Utiliza siempre argumentos precisos basados en la consulta del usuario.\n"
    "- Reporta de forma clara y transparente el resultado de cada acción al usuario gastronómico."
)

# Estructura estricta exigida por R2:
# (1) "system", (2) chat_history, (3) human {entrada}, (4) agent_scratchpad
prompt_agente_horeca: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", PROMPT_SISTEMA_HORECA),
    MessagesPlaceholder("chat_history"),
    ("human", "{entrada}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# ============================================================================
# 5. CONSTRUCCIÓN DEL AGENTE, EXECUTOR Y MEMORIA CONVERSACIONAL (R1 & R2)
# ============================================================================

def construir_modelo_llm(modelo: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Inicializa ChatGoogleGenerativeAI a partir de variables de entorno con fallback defensivo."""
    load_dotenv(override=True)
    modelo_seleccionado = modelo or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
    clave_api = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or "CLAVE_MOCK_SANDBOX"
    )
    return ChatGoogleGenerativeAI(
        model=modelo_seleccionado,
        google_api_key=clave_api,
    )


def construir_agente_horeca(
    modelo_llm: Optional[Any] = None,
    herramientas: Optional[List[Any]] = None,
    prompt: Optional[ChatPromptTemplate] = None,
) -> Any:
    """Construye el agente de llamadas a herramientas usando create_tool_calling_agent."""
    if modelo_llm is None:
        modelo_llm = construir_modelo_llm()
    if herramientas is None:
        herramientas = HERRAMIENTAS_HORECA
    if prompt is None:
        prompt = prompt_agente_horeca
    return create_tool_calling_agent(modelo_llm, herramientas, prompt)


def construir_ejecutor_agente(
    agente: Optional[Any] = None,
    herramientas: Optional[List[Any]] = None,
    modelo_llm: Optional[Any] = None,
    return_intermediate_steps: bool = True,
    verbose: bool = False,
) -> AgentExecutor:
    """Construye el AgentExecutor con return_intermediate_steps=True de forma obligatoria."""
    if herramientas is None:
        herramientas = HERRAMIENTAS_HORECA
    if agente is None:
        agente = construir_agente_horeca(modelo_llm=modelo_llm, herramientas=herramientas)
    return AgentExecutor(
        agent=agente,
        tools=herramientas,
        return_intermediate_steps=return_intermediate_steps,
        verbose=verbose,
    )


def obtener_historial_sesion(session_id: str) -> InMemoryChatMessageHistory:
    """Retorna o inicializa el historial de mensajes estrictamente aislado por session_id."""
    id_clave = str(session_id) if session_id is not None else "sesion_predeterminada"
    if id_clave not in almacen_historiales:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            almacen_historiales[id_clave] = InMemoryChatMessageHistory()
    return almacen_historiales[id_clave]


def limpiar_historial_sesiones() -> None:
    """Limpia el diccionario de historiales en memoria (utilizado para pruebas limpias)."""
    almacen_historiales.clear()


def limpiar_registros_transaccionales() -> None:
    """Limpia los registros transaccionales en memoria (utilizado para pruebas limpias)."""
    BORRADORES_PEDIDOS.clear()
    ORDENES_COMPRA_EMITIDAS.clear()
    PAGOS_PROCESADOS.clear()


def _normalizar_contenido_respuesta(contenido: Any) -> str:
    """Convierte respuestas en lista de bloques (ej. Gemini thinking/text) a str plano."""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes: List[str] = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type") == "text":
                texto = bloque.get("text")
                if isinstance(texto, str):
                    partes.append(texto)
        if partes:
            return "\n".join(partes)
    return "" if contenido is None else str(contenido)


def _normalizar_salida_ejecutor(resultado: Any) -> Any:
    """Normaliza la clave 'output' del AgentExecutor a str antes de guardarla en el historial."""
    if isinstance(resultado, dict) and "output" in resultado:
        return {
            **resultado,
            "output": _normalizar_contenido_respuesta(resultado.get("output")),
        }
    return resultado


def construir_agente_con_historial(
    ejecutor: Optional[AgentExecutor] = None,
    modelo_llm: Optional[Any] = None,
) -> RunnableWithMessageHistory:
    """Envuelve el AgentExecutor con RunnableWithMessageHistory garantizando memoria por sesión."""
    if ejecutor is None:
        ejecutor = construir_ejecutor_agente(modelo_llm=modelo_llm)
    ejecutor_normalizado = ejecutor | RunnableLambda(_normalizar_salida_ejecutor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return RunnableWithMessageHistory(
            runnable=ejecutor_normalizado,
            get_session_history=obtener_historial_sesion,
            input_messages_key="entrada",
            history_messages_key="chat_history",
            output_messages_key="output",
        )


def ejecutar_agente(
    entrada: Optional[str] = "",
    session_id: str = "sesion_predeterminada",
    agente_ejecutable: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ejecuta una interacción con el agente preservando la sesión y trazabilidad completa."""
    if agente_ejecutable is None:
        agente_ejecutable = agente_con_historial

    texto_entrada = "" if entrada is None else str(entrada)
    id_sesion = str(session_id) if session_id is not None else "sesion_predeterminada"
    token_sesion = sesion_actual.set(id_sesion)
    try:
        resultado = agente_ejecutable.invoke(
            {"entrada": texto_entrada},
            config={"configurable": {"session_id": id_sesion}},
        )
        return resultado
    finally:
        sesion_actual.reset(token_sesion)


# Instancias por defecto expuestas a nivel de módulo
modelo_llm_defecto = construir_modelo_llm()
agente_defecto = construir_agente_horeca(modelo_llm=modelo_llm_defecto)
ejecutor_agente = construir_ejecutor_agente(agente=agente_defecto)
agente_con_historial = construir_agente_con_historial(ejecutor=ejecutor_agente)


# ============================================================================
# 6. ENTRADA INTERACTIVA DE CONSOLA PARA PRUEBAS MANUALES EN VIVO
# ============================================================================

def iniciar_interfaz_consola() -> None:
    """Inicia la interfaz de línea de comando interactiva para generar trazas reales."""
    print("=" * 72)
    print("🍽️  HORECA AI Service — Agente Inteligente con Herramientas y Autonomía")
    print("=" * 72)
    print("Herramientas gastronómicas configuradas:")
    print("  [Riesgo BAJO]  consultar_catalogo_insumos, calcular_cotizacion")
    print("  [Riesgo MEDIO] crear_borrador_pedido, cancelar_borrador_pedido")
    print("  [Riesgo ALTO]  emitir_orden_compra_formal, procesar_pago_proveedor (Requieren HITL)")
    print(f"Auditoría activa en: '{ARCHIVO_TRAZAS_DEFECTO}' y '{ARCHIVO_APROBACIONES_DEFECTO}'.")
    print("Escriba 'salir' o 'exit' para concluir la sesión.\n")

    id_sesion = f"sesion_{int(datetime.now().timestamp())}"
    print(f"Sesión iniciada con identificador: {id_sesion}\n")

    while True:
        try:
            texto_usuario = input("HORECA AI Service > ").strip()
            if not texto_usuario:
                continue
            if texto_usuario.lower() in ("salir", "exit", "quit"):
                print("Finalizando sesión interactiva. ¡Buen servicio gastronómico!")
                break

            respuesta = ejecutar_agente(texto_usuario, session_id=id_sesion)
            salida = respuesta.get("output", "")
            pasos_intermedios = respuesta.get("intermediate_steps", [])

            print(f"\nAsistente: {salida}\n")
            if pasos_intermedios:
                print(f"[Auditoría]: Se ejecutaron {len(pasos_intermedios)} herramientas.")
                for paso in pasos_intermedios:
                    accion, obs = paso
                    print(f"  - Herramienta: {accion.tool} | Entrada: {accion.tool_input}")

        except (KeyboardInterrupt, EOFError):
            print("\nSesión cancelada por el usuario.")
            break
        except Exception as error_bucle:
            print(f"\n[ERROR EN EJECUCIÓN] {error_bucle}\n")


if __name__ == "__main__":
    iniciar_interfaz_consola()
