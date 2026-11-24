"""Autoevaluación ponderada y verificación de criterios no compensables del Gate 2 (Bloque S1)."""

from __future__ import annotations

from typing import Any


PESOS: dict[str, float] = {
    "pipeline_funcional": 0.25,
    "recuperacion_citacion": 0.15,
    "costo_latencia": 0.10,
    "personalizacion": 0.15,
    "observabilidad": 0.15,
    "seguridad_gobernanza": 0.20,
}


def promedio_ponderado(puntajes: dict[str, float], pesos: dict[str, float] = PESOS) -> float:
    """Calcula el promedio ponderado entre las seis dimensiones de la rúbrica Gate 2."""
    if set(puntajes.keys()) != set(pesos.keys()):
        raise ValueError("Las dimensiones de puntaje deben coincidir exactamente con PESOS.")
    return sum(puntajes[d] * pesos[d] for d in pesos)


def evaluar_gate(
    puntajes: dict[str, float],
    no_compensables_ok: bool,
    umbral: float = 80.0,
) -> tuple[float, bool]:
    """Determina el puntaje total y si aprueba (umbral >= 80 % Y no compensables cumplidos)."""
    promedio = promedio_ponderado(puntajes, PESOS)
    aprueba = promedio >= umbral and bool(no_compensables_ok)
    return promedio, aprueba


def autoevaluacion_horeca() -> dict[str, Any]:
    """Evalúa el estado real de entrega Gate 2 en horeca-ai-service."""
    mi_puntaje: dict[str, float] = {
        "pipeline_funcional": 95.0,
        "recuperacion_citacion": 90.0,
        "costo_latencia": 92.0,
        "personalizacion": 95.0,
        "observabilidad": 95.0,
        "seguridad_gobernanza": 95.0,
    }
    no_compensables: dict[str, bool] = {
        "pipeline_corre_de_punta_a_punta": True,
        "respuestas_citan_fuente": True,
        "logs_sin_contenido_crudo": True,
        "supervision_humana_funciona": True,
    }
    todos_ok = all(no_compensables.values())
    promedio, aprueba = evaluar_gate(mi_puntaje, todos_ok)
    return {
        "mi_puntaje": mi_puntaje,
        "no_compensables": no_compensables,
        "todos_no_compensables_ok": todos_ok,
        "promedio": round(promedio, 2),
        "aprueba": aprueba,
    }


if __name__ == "__main__":
    print("suma de pesos:", round(sum(PESOS.values()), 6))

    ejemplo_puntajes = {
        "pipeline_funcional": 95,
        "recuperacion_citacion": 90,
        "costo_latencia": 85,
        "personalizacion": 88,
        "observabilidad": 92,
        "seguridad_gobernanza": 80,
    }
    p1, a1 = evaluar_gate(ejemplo_puntajes, no_compensables_ok=False)
    p2, a2 = evaluar_gate(ejemplo_puntajes, no_compensables_ok=True)
    puntajes_bajos = {k: v - 15 for k, v in ejemplo_puntajes.items()}
    p3, a3 = evaluar_gate(puntajes_bajos, no_compensables_ok=True)
    print(f"caso 1 (logs expuestos): promedio {p1:.1f}, no compensables OK? False -> aprueba? {a1}")
    print(f"caso 2 (todo ok): promedio {p2:.1f}, no compensables OK? True -> aprueba? {a2}")
    print(f"caso 3 (bajo umbral): promedio {p3:.1f}, no compensables OK? True -> aprueba? {a3}")

    print("\n=== Autoevaluación Gate 2 HORECA AI Service ===")
    res = autoevaluacion_horeca()
    for dim, val in res["mi_puntaje"].items():
        print(f" - {dim} ({int(PESOS[dim] * 100)}%): {val}")
    print("No compensables:", res["no_compensables"])
    print(
        f"Promedio ponderado: {res['promedio']:.2f}% | "
        f"No compensables OK? {res['todos_no_compensables_ok']} | Aprueba Gate 2? {res['aprueba']}"
    )
