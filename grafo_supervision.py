"""Grafo de supervisión humana (HITL) para despliegue de prompts HORECA (Unidad 2.7 - Bloque 32 Paso 2)."""

from __future__ import annotations

import logging
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


logger = logging.getLogger("horeca.supervision")


class EstadoDespliegue(TypedDict):
    """Estado del flujo de aprobación humana antes de publicar una versión de prompt."""

    huella_nueva: str
    aprobado: bool


def proponer(state: EstadoDespliegue) -> dict[str, Any]:
    """Prepara la propuesta de despliegue registrando la huella del prompt candidato."""
    logger.info("Propuesta de despliegue iniciada para huella=%s", state["huella_nueva"])
    return {}


def supervisar(state: EstadoDespliegue) -> dict[str, Any]:
    """Detiene el grafo con interrupt() para exigir aprobación humana explícita."""
    decision = interrupt(
        {
            "motivo": "Aprobar despliegue de prompt en producción HORECA AI Service",
            "huella_propuesta": state["huella_nueva"],
        }
    )
    return {"aprobado": bool(decision)}


def aplicar(state: EstadoDespliegue) -> dict[str, Any]:
    """Ejecuta o cancela la publicación en función de la decisión humana."""
    if state["aprobado"]:
        print(f"Desplegando versión de prompt HORECA {state['huella_nueva']}")
    else:
        print(f"Despliegue rechazado para versión de prompt {state['huella_nueva']}; se conserva versión vigente")
    return {}


def construir_grafo_supervision(checkpointer: Any | None = None):
    """Compila el StateGraph de supervisión con checkpointer para soportar interrupt/resume."""
    grafo = StateGraph(EstadoDespliegue)
    grafo.add_node("proponer", proponer)
    grafo.add_node("supervisar", supervisar)
    grafo.add_node("aplicar", aplicar)

    grafo.add_edge(START, "proponer")
    grafo.add_edge("proponer", "supervisar")
    grafo.add_edge("supervisar", "aplicar")
    grafo.add_edge("aplicar", END)

    return grafo.compile(checkpointer=checkpointer or InMemorySaver())


app_despliegue = construir_grafo_supervision()


def nuevo_thread_id(prefijo: str = "despliegue-horeca") -> str:
    """Genera un identificador único por solicitud de aprobación para evitar colisiones."""
    return f"{prefijo}-{uuid.uuid4().hex[:10]}"


def iniciar_despliegue(
    huella_nueva: str,
    app: Any | None = None,
    thread_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inicia una propuesta de despliegue y retorna (resultado_parcial, config_con_thread_id)."""
    instancia = app or app_despliegue
    tid = thread_id or nuevo_thread_id()
    config = {"configurable": {"thread_id": tid}}
    salida = instancia.invoke(
        {"huella_nueva": huella_nueva, "aprobado": False},
        config=config,
    )
    return salida, config


def resolver(app: Any, config: dict[str, Any], aprobar: bool) -> dict[str, Any]:
    """Reanuda un despliegue en pausa validando antes que siga pendiente en supervise."""
    snapshot = app.get_state(config)
    if not snapshot.next:
        raise ValueError(
            "Este despliegue ya finalizó o fue cancelado; crea un nuevo thread_id para proponer otro."
        )
    return app.invoke(Command(resume=bool(aprobar)), config=config)


if __name__ == "__main__":
    from registro_prompts import construir_registro_horeca

    registro = construir_registro_horeca()
    huella_v2 = registro.ultima("rag_proveedores_horeca")["huella"]

    print("=== Flujo 1: Pausa en supervisar + Aprobación (Command(resume=True)) ===")
    parcial_1, cfg_aprobar = iniciar_despliegue(huella_v2, app=app_despliegue)
    estado_pausa = app_despliegue.get_state(cfg_aprobar)
    print(f"__interrupt__ = {parcial_1.get('__interrupt__')}")
    print(f"get_state.next = {estado_pausa.next}")

    final_aprobado = resolver(app_despliegue, cfg_aprobar, aprobar=True)
    estado_final_1 = app_despliegue.get_state(cfg_aprobar)
    print(f"Resultado aprobado = {final_aprobado} | next tras resume = {estado_final_1.next}")

    print("\n=== Flujo 2: Pausa en supervisar + Rechazo (Command(resume=False)) en otro thread_id ===")
    parcial_2, cfg_rechazar = iniciar_despliegue(huella_v2, app=app_despliegue)
    final_rechazado = resolver(app_despliegue, cfg_rechazar, aprobar=False)
    estado_final_2 = app_despliegue.get_state(cfg_rechazar)
    print(f"Resultado rechazado = {final_rechazado} | next tras rechazo = {estado_final_2.next}")

    try:
        resolver(app_despliegue, cfg_rechazar, aprobar=True)
    except ValueError as err:
        print(f"\nGuardia doble-resume verificada: {err}")
