"""
decision.py - Método cuantitativo de decisión y estimación de costos de personalización (Bloque 30 / U2.6).

Aplica el modelo puntuado del Bloque 30 (RAG vs Fine-tuning completo vs PEFT) con detección
explícita de empates, análisis de sensibilidad por pregunta, cálculo del costo humano de
etiquetado y generación del esqueleto de decision_personalizacion.md para HORECA AI Service.
"""

from typing import Any, Dict, List, Tuple


# Respuestas evaluadas sobre evidencia verificable del proyecto horeca-ai-service:
# - Catálogo de insumos y documentación indexada cambian frecuentemente (precios/stock/proveedores).
# - Consistencia JSON ya verificada al 100% en prompts/registro_resultados.json (9/9) + parseo robusto.
# - Dataset actual de tuning tiene 12 ejemplos (< 200 requeridos).
# - Operación con Free Gemini API Key (límites 429) e iteraciones rápidas por Gate.
# - Plataforma única B2B HORECA que atiende sesiones aisladas sin requerir modelos separados.
respuestas_horeca: Dict[str, bool] = {
    "informacion_cambia_seguido": True,
    "problema_de_formato_o_tono": False,
    "dataset_grande_y_de_calidad": False,
    "presupuesto_limitado_o_iterar_rapido": True,
    "necesita_multiples_variantes": False,
}

PROBLEMA_HORECA: str = (
    "Al responder consultas entre negocios del sector HORECA y proveedores mayoristas, "
    "el servicio necesita reflejar cambios frecuentes en precios, disponibilidad de stock, "
    "especificaciones de envase y alérgenos sin inventar datos, manteniendo formato y costos acotados."
)


def estimar_costo_de_etiquetar(
    num_ejemplos: float,
    minutos_por_ejemplo: float,
    costo_por_hora: float,
) -> Tuple[float, float]:
    """Estima horas humanas y costo monetario de etiquetar un dataset de entrenamiento."""
    horas = (num_ejemplos * minutos_por_ejemplo) / 60
    costo = horas * costo_por_hora
    return horas, costo


def recomendar_estrategia(
    respuestas: Dict[str, bool],
) -> Tuple[Dict[str, int], List[str]]:
    """Puntúa RAG, Fine-tuning completo y PEFT conservando empates sin ocultarlos."""
    if any(valor is None for valor in respuestas.values()):
        raise SystemExit("Completa las cinco respuestas con True o False antes de continuar.")

    puntos: Dict[str, int] = {"RAG": 0, "Fine-tuning completo": 0, "PEFT": 0}
    if respuestas["informacion_cambia_seguido"]:
        puntos["RAG"] += 3
    if respuestas["problema_de_formato_o_tono"]:
        puntos["PEFT"] += 2
        puntos["Fine-tuning completo"] += 1
    if respuestas["dataset_grande_y_de_calidad"]:
        puntos["Fine-tuning completo"] += 2
        puntos["PEFT"] += 1
    if respuestas["presupuesto_limitado_o_iterar_rapido"]:
        puntos["RAG"] += 2
        puntos["PEFT"] += 1
    if respuestas["necesita_multiples_variantes"]:
        puntos["PEFT"] += 2

    maximo = max(puntos.values())
    ganadoras = [tecnica for tecnica, puntaje in puntos.items() if puntaje == maximo]
    return puntos, ganadoras


def analizar_sensibilidad(
    respuestas: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """Recalcula la recomendación invirtiendo una respuesta a la vez para medir estabilidad."""
    _, ganadoras_base = recomendar_estrategia(respuestas)
    analisis: List[Dict[str, Any]] = []
    for clave in respuestas:
        variante = dict(respuestas)
        variante[clave] = not variante[clave]
        puntos_v, ganadoras_v = recomendar_estrategia(variante)
        analisis.append({
            "clave": clave,
            "nuevo_valor": variante[clave],
            "puntos": puntos_v,
            "ganadoras": ganadoras_v,
            "cambia": set(ganadoras_v) != set(ganadoras_base),
        })
    return analisis


def generar_esqueleto_decision(
    problema: str,
    puntos: Dict[str, int],
    ganadoras: List[str],
) -> str:
    """Genera el esqueleto de 4 secciones para el documento de decisión fundamentada."""
    empate = len(ganadoras) > 1
    lineas: List[str] = [
        "# Decision de personalizacion",
        "",
        "## 1. El problema",
        problema,
        "",
        "## 2. Resultado del metodo de decision",
    ]
    for tecnica, p in sorted(puntos.items(), key=lambda x: -x[1]):
        lineas.append(f"- {tecnica}: {p} puntos")
    if empate:
        lineas.append("")
        lineas.append(f"**Empate entre: {', '.join(ganadoras)}**")
    lineas.append("")
    lineas.append("## 3. Tecnica elegida y por que")
    if empate:
        lineas.append(
            "[COMPLETAR: el puntaje no resolvio el empate solo -- "
            "justifica aqui con criterio propio]"
        )
    else:
        lineas.append(f"[COMPLETAR: por que {ganadoras[0]} responde mejor a este caso]")
    lineas.append("")
    lineas.append("## 4. Que necesitarias para ejecutarla")
    lineas.append(
        "[COMPLETAR: corpus faltante, o cantidad de ejemplos etiquetados que faltan]"
    )
    return "\n".join(lineas)


if __name__ == "__main__":
    puntos_horeca, ganadoras_horeca = recomendar_estrategia(respuestas_horeca)
    print("=== Evaluación HORECA AI Service ===")
    print("Respuestas:", respuestas_horeca)
    print("Puntaje:", puntos_horeca)
    print("Candidatas:", ganadoras_horeca)

    print("\n=== Análisis de sensibilidad (cambiando una respuesta a la vez) ===")
    for fila in analizar_sensibilidad(respuestas_horeca):
        etiqueta = "CAMBIA" if fila["cambia"] else "igual"
        print(
            f"{fila['clave']} -> {fila['nuevo_valor']}: "
            f"{fila['ganadoras']} ({fila['puntos']}) [{etiqueta}]"
        )

    horas_base, costo_base = estimar_costo_de_etiquetar(188, 5, 25)
    horas_rev, costo_rev = estimar_costo_de_etiquetar(188 * 0.20, 2, 25)
    total_horas = horas_base + horas_rev
    total_costo = costo_base + costo_rev
    print("\n=== Estimación si se ampliara dataset PEFT (de 12 a 200 ejemplos: faltan 188) ===")
    print(f"Etiquetado base (188 ej x 5 min): {horas_base:.2f} horas, aprox ${costo_base:.2f}")
    print(f"Segunda revisión (20% x 2 min):   {horas_rev:.2f} horas, aprox ${costo_rev:.2f}")
    print(f"Total: {total_horas:.2f} horas ({total_horas / 8:.1f} jornadas de 8h), aprox ${total_costo:.2f}")
