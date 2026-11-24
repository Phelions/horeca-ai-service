"""Matriz de riesgos operativos y gobernanza técnica HORECA AI Service (Unidad 2.7 - Bloque 32 Paso 3)."""

from __future__ import annotations

from typing import Any


def puntaje_riesgo(probabilidad: int, impacto: int) -> int:
    """Calcula el puntaje de riesgo (1..25) validando rango 1..5 para probabilidad e impacto."""
    if not (1 <= probabilidad <= 5) or not (1 <= impacto <= 5):
        raise ValueError("Probabilidad e impacto deben estar en el rango entero 1..5.")
    return probabilidad * impacto


def categoria_riesgo(puntaje: int) -> str:
    """Clasifica el puntaje en CRITICO (>=15), ALTO (>=9), MEDIO (>=4) o BAJO (<4)."""
    if puntaje >= 15:
        return "CRITICO"
    if puntaje >= 9:
        return "ALTO"
    if puntaje >= 4:
        return "MEDIO"
    return "BAJO"


def esta_en_el_limite(probabilidad: int, impacto: int) -> bool:
    """Indica si variar en +/-1 probabilidad o impacto cambia la categoría del riesgo."""
    base = categoria_riesgo(puntaje_riesgo(probabilidad, impacto))
    for dp in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dp == 0 and di == 0:
                continue
            p = max(1, min(5, probabilidad + dp))
            i = max(1, min(5, impacto + di))
            if categoria_riesgo(puntaje_riesgo(p, i)) != base:
                return True
    return False


mis_riesgos: list[dict[str, Any]] = [
    {
        "id": "R-01",
        "riesgo": (
            "Inyección indirecta de instrucciones ocultas en fichas técnicas o catálogos "
            "cargados por proveedores mayoristas (ej. alteración forzada de descuentos o plazos)"
        ),
        "probabilidad": 3,
        "impacto": 5,
        "mitigacion": (
            "Bloque 31 (logging_rag.py): defensa multicapa con separación SystemMessage/HumanMessage, "
            "delimitadores <contexto_recuperado> y filtro regex filtrar_documentos_sospechosos "
            "que registra documento_descartado_por_inyeccion."
        ),
        "bloques_referencia": ["Bloque 31", "Bloque 32"],
        "punto_supervision": (
            "Cuarentena manual por operador B2B ante evento documento_descartado_por_inyeccion "
            "antes de reindexar al proveedor + aprobación HITL en grafo_supervision.py al actualizar reglas."
        ),
    },
    {
        "id": "R-02",
        "riesgo": (
            "Ejecución autónoma de pedidos mayoristas, bloqueos de crédito o cambios de lista de "
            "precios B2B por el agente con herramientas sin consentimiento explícito del comprador HORECA"
        ),
        "probabilidad": 4,
        "impacto": 5,
        "mitigacion": (
            "Bloque 23 (agente_con_herramientas.py): clasificación en tres niveles de riesgo "
            "(bajo, medio, alto) con pausa obligatoria esperar_aprobacion para acciones irreversibles."
        ),
        "bloques_referencia": ["Bloque 23"],
        "punto_supervision": (
            "Compuerta humana síncrona esperar_aprobacion (Bloque 23) mostrando proveedor, SKU, "
            "monto total y ventana de despacho antes de confirmar la orden."
        ),
    },
    {
        "id": "R-03",
        "riesgo": (
            "Alucinación o mezcla de condiciones comerciales (pedido mínimo, temperatura de cadena de frío, "
            "tiempos de entrega) entre proveedores mayoristas en respuestas RAG"
        ),
        "probabilidad": 3,
        "impacto": 3,
        "mitigacion": (
            "Bloque 25 (indexar_corpus.py), Bloque 26 (recuperacion.py) y Bloque 27 (evaluacion_rag.py): "
            "citas numeradas obligatorias [1], instrucción explícita de no inventar y evaluación de "
            "precisión/exhaustividad de la recuperación."
        ),
        "bloques_referencia": ["Bloque 25", "Bloque 26", "Bloque 27", "Bloque 32"],
        "punto_supervision": (
            "Revisión y aprobación humana en grafo_supervision.py (Bloque 32) antes de publicar "
            "cualquier actualización de PROMPT_HORECA_RAG_V2 o parámetros de recuperación en producción."
        ),
    },
    {
        "id": "R-04",
        "riesgo": (
            "Filtración de precios privados negociados, volúmenes de compra o identificadores de "
            "clientes HORECA en logs de aplicación o trazas de callbacks"
        ),
        "probabilidad": 2,
        "impacto": 5,
        "mitigacion": (
            "Bloque 31 (logging_rag.py): sustitución de texto crudo por huella(texto) SHA-256 de 12 hex "
            "y pregunta_len en logs JSON correlacionados únicamente por trace_id."
        ),
        "bloques_referencia": ["Bloque 31", "Bloque 32"],
        "punto_supervision": (
            "Auditoría técnica y aprobación en grafo_supervision.py (Bloque 32) ante cualquier cambio "
            "en HandlerDeTrazasRAG o campos emitidos por JsonFormatter."
        ),
    },
    {
        "id": "R-05",
        "riesgo": (
            "Desbalance de categorías o sesgo de recomendación hacia pocos proveedores mayoristas "
            "si se entrena un modelo personalizado con datos históricos sesgados"
        ),
        "probabilidad": 3,
        "impacto": 4,
        "mitigacion": (
            "Bloque 29 (dataset_tuning.py) y Bloque 30 (laboratorio_personalizacion.py): validación "
            "automática de esquema, duplicados y distribución mediante chequear_balance() antes de "
            "entrenar LoRA, priorizando RAG para catálogo y stock dinámico."
        ),
        "bloques_referencia": ["Bloque 29", "Bloque 30", "Bloque 32"],
        "punto_supervision": (
            "Revisión de reporte chequear_balance() y compuerta humana grafo_supervision.py (Bloque 32) "
            "antes de promover cualquier adaptador PEFT o cambio de estrategia en producción."
        ),
    },
    {
        "id": "R-06",
        "riesgo": (
            "Incremento abrupto de latencia o costo de tokens por consultas repetitivas en horarios "
            "pico de reposición semanal de restaurantes y hoteles"
        ),
        "probabilidad": 2,
        "impacto": 3,
        "mitigacion": (
            "Bloque 28 (registro_rag.py y benchmark_rag.py) y Bloque 31 (registrar_consumo): "
            "registro estructurado de latencia_ms, costo_usd e input/output/total tokens por trace_id."
        ),
        "bloques_referencia": ["Bloque 28", "Bloque 31"],
        "punto_supervision": (
            "Monitoreo semanal de umbrales de consumo por operaciones sin bloqueo síncrono al usuario."
        ),
    },
    {
        "id": "R-07",
        "riesgo": (
            "Variación menor de estilo o formato en sugerencias generales de estacionalidad "
            "culinaria sin impacto en precios, inventario ni despacho"
        ),
        "probabilidad": 1,
        "impacto": 2,
        "mitigacion": (
            "Pruebas de regresión de esquema y formato de citas en suite pytest automatizada."
        ),
        "bloques_referencia": ["Bloque 26"],
        "punto_supervision": "No aplica supervisión síncrona; revisión en muestreo de calidad.",
    },
]


def evaluar_matriz(riesgos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enriquece cada riesgo con puntaje, categoría y bandera esta_en_el_limite."""
    evaluados: list[dict[str, Any]] = []
    for item in riesgos:
        puntaje = puntaje_riesgo(item["probabilidad"], item["impacto"])
        categoria = categoria_riesgo(puntaje)
        en_limite = esta_en_el_limite(item["probabilidad"], item["impacto"])
        evaluados.append(
            {
                **item,
                "puntaje": puntaje,
                "categoria": categoria,
                "en_el_limite": en_limite,
            }
        )
    return evaluados


def validar_matriz(riesgos: list[dict[str, Any]]) -> dict[str, Any]:
    """Verifica criterios mínimos de gobernanza (Bloque 32): cantidad, variación y supervisión."""
    evaluados = evaluar_matriz(riesgos)
    if len(evaluados) < 5:
        raise ValueError("La matriz debe incluir al menos 5 riesgos propios del proyecto.")

    pares = {(r["probabilidad"], r["impacto"]) for r in evaluados}
    if len(pares) < 3:
        raise ValueError("La matriz debe diferenciar probabilidades e impactos entre riesgos.")

    criticos_o_altos = [r for r in evaluados if r["categoria"] in {"CRITICO", "ALTO"}]
    if not criticos_o_altos:
        raise ValueError("La matriz debe identificar al menos un riesgo ALTO o CRITICO.")

    for r in criticos_o_altos:
        if not str(r.get("punto_supervision", "")).strip():
            raise ValueError(
                f"El riesgo {r.get('id')} ({r['categoria']}) requiere punto_supervision explícito."
            )

    por_categoria: dict[str, int] = {}
    for r in evaluados:
        por_categoria[r["categoria"]] = por_categoria.get(r["categoria"], 0) + 1

    return {
        "total": len(evaluados),
        "por_categoria": por_categoria,
        "en_el_limite": [r["id"] for r in evaluados if r["en_el_limite"]],
    }


if __name__ == "__main__":
    resumen = validar_matriz(mis_riesgos)
    evaluados = evaluar_matriz(mis_riesgos)

    print("=== Matriz de Riesgos Técnicos HORECA AI Service ===")
    for r in evaluados:
        marca = " [AL LIMITE]" if r["en_el_limite"] else ""
        print(
            f"{r['id']} [{r['categoria']}{marca}] "
            f"p={r['probabilidad']} x i={r['impacto']} = {r['puntaje']} -> {r['riesgo']}"
        )
        print(f"   Mitigación: {r['mitigacion']}")
        if r["categoria"] in {"CRITICO", "ALTO"}:
            print(f"   Supervisión humana: {r['punto_supervision']}")

    print("\n--- Chequeo de contraste en frontera ---")
    print("p=3, i=5 (15):", categoria_riesgo(puntaje_riesgo(3, 5)), "| limite:", esta_en_el_limite(3, 5))
    print("p=2, i=5 (10):", categoria_riesgo(puntaje_riesgo(2, 5)), "| limite:", esta_en_el_limite(2, 5))
    print("p=5, i=5 (25):", categoria_riesgo(puntaje_riesgo(5, 5)), "| limite:", esta_en_el_limite(5, 5))
    print(f"\nResumen validación: {resumen}")
