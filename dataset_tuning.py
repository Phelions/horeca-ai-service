"""
dataset_tuning.py - Laboratorio guiado de dataset para personalización PEFT (Bloque 29 / U2.6).

Prepara, balancea y valida un dataset propio de la plataforma B2B HORECA AI Service
(que conecta negocios del sector HORECA con proveedores mayoristas) para normalizar
mensajes entrantes de texto libre hacia un JSON con esquema fijo ('categoria' y 'prioridad').
"""

import json
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types


CLAVES_ESPERADAS_HORECA: Tuple[str, ...] = ("categoria", "prioridad")

# 12 ejemplos propios del dominio B2B HORECA (4 categorías balanceadas x 3 ejemplos c/u)
ejemplos_raw: List[Tuple[str, Dict[str, str]]] = [
    (
        "Necesitamos urgente para mañana a las 07:00 10 sacos de harina fuerza 25 kg de Molino Central para el hotel.",
        {"categoria": "pedido_compra", "prioridad": "alta"},
    ),
    (
        "Por favor agendar reposición semanal de 6 bidones de aceite alto oleico 20L para la próxima semana.",
        {"categoria": "pedido_compra", "prioridad": "media"},
    ),
    (
        "Queremos sumar 15 kilos de queso mozzarella en barra y 8 cajas de tomate triturado al pedido habitual.",
        {"categoria": "pedido_compra", "prioridad": "media"},
    ),
    (
        "El despacho de Frigorífico Andino llegó sin cadena de frío y con 2 cajas de lomo abiertas, urge reemplazo.",
        {"categoria": "reclamo_despacho", "prioridad": "alta"},
    ),
    (
        "Pedimos aceite alto oleico de 20L pero el proveedor entregó aceite vegetal común en bidones de 10L.",
        {"categoria": "reclamo_despacho", "prioridad": "alta"},
    ),
    (
        "En la factura #4821 de Agrícola El Sol faltaron 3 latas de tomate respecto a la guía recibida en bodega.",
        {"categoria": "reclamo_despacho", "prioridad": "media"},
    ),
    (
        "¿El queso mozzarella laminado que distribuye Lácteos del Sur está certificado libre de gluten?",
        {"categoria": "consulta_catalogo", "prioridad": "media"},
    ),
    (
        "Cuando puedan, ¿me envían la ficha de formatos y precios mayoristas de arroz grano largo grado 1?",
        {"categoria": "consulta_catalogo", "prioridad": "baja"},
    ),
    (
        "Quisiéramos saber si algún proveedor del portal despacha harina integral orgánica los sábados.",
        {"categoria": "consulta_catalogo", "prioridad": "baja"},
    ),
    (
        "Anular de inmediato el borrador BORR-0042 porque ya cubrimos ese stock con otro distribuidor.",
        {"categoria": "ajuste_o_cancelacion", "prioridad": "alta"},
    ),
    (
        "Mover la ventana de recepción del borrador BORR-0039 de las 08:00 a las 11:00 para el turno de tarde.",
        {"categoria": "ajuste_o_cancelacion", "prioridad": "media"},
    ),
    (
        "Bajar de 12 a 8 kilos el pollo del borrador de la próxima semana si todavía no fue emitido.",
        {"categoria": "ajuste_o_cancelacion", "prioridad": "baja"},
    ),
]


def construir_ejemplos_tuning(
    datos_crudos: List[Tuple[str, Dict[str, str]]] = ejemplos_raw,
) -> List[types.TuningExample]:
    """Convierte pares (texto, dict_salida) en objetos types.TuningExample del SDK."""
    return [
        types.TuningExample(
            text_input=texto,
            output=json.dumps(salida, ensure_ascii=False),
        )
        for texto, salida in datos_crudos
    ]


examples: List[types.TuningExample] = construir_ejemplos_tuning(ejemplos_raw)


def chequear_balance(
    datos_crudos: List[Tuple[str, Dict[str, str]]],
    clave: str = "categoria",
) -> Counter:
    """Cuenta cuántos ejemplos existen por categoría para detectar sesgos de representación."""
    return Counter(salida[clave] for _, salida in datos_crudos)


def validar_dataset(
    ejemplos: List[types.TuningExample],
    claves_esperadas: Tuple[str, ...] = CLAVES_ESPERADAS_HORECA,
) -> List[str]:
    """Valida entradas no vacías, ausencia de duplicados y salidas JSON válidas con claves esperadas."""
    problemas: List[str] = []
    vistos = set()
    for i, ej in enumerate(ejemplos):
        if not ej.text_input or not ej.text_input.strip():
            problemas.append(f"ejemplo {i}: text_input vacio")
        if not ej.output or not ej.output.strip():
            problemas.append(f"ejemplo {i}: output vacio")
            continue
        if ej.text_input in vistos:
            problemas.append(f"ejemplo {i}: text_input duplicado")
        vistos.add(ej.text_input)
        try:
            salida = json.loads(ej.output)
        except json.JSONDecodeError:
            problemas.append(f"ejemplo {i}: output no es JSON valido")
            continue
        if not isinstance(salida, dict):
            problemas.append(f"ejemplo {i}: output JSON debe ser un objeto")
            continue
        for clave in claves_esperadas:
            if clave not in salida:
                problemas.append(f'ejemplo {i}: output no tiene la clave "{clave}"')
    return problemas


def construir_configuracion_peft(
    ejemplos: List[types.TuningExample] = examples,
    nombre_modelo_mostrado: str = "horeca-b2b-clasificador-v1",
    epoch_count: int = 5,
) -> Tuple[types.TuningDataset, types.CreateTuningJobConfig]:
    """Construye los objetos TuningDataset y CreateTuningJobConfig para un trabajo PEFT."""
    dataset = types.TuningDataset(examples=ejemplos)
    config = types.CreateTuningJobConfig(
        tuning_mode=types.TuningMode.TUNING_MODE_PEFT_ADAPTER,
        adapter_size=types.AdapterSize.ADAPTER_SIZE_FOUR,
        epoch_count=epoch_count,
        tuned_model_display_name=nombre_modelo_mostrado,
    )
    return dataset, config


def esperar_a_que_termine(
    client: genai.Client,
    nombre_trabajo: str,
    intervalo_segundos: int = 30,
) -> types.TuningJob:
    """Consulta periódicamente un trabajo de tuning asíncrono hasta alcanzar un estado terminal."""
    estados_finales = {
        types.JobState.JOB_STATE_SUCCEEDED,
        types.JobState.JOB_STATE_FAILED,
        types.JobState.JOB_STATE_CANCELLED,
        types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
    }
    while True:
        trabajo: Any = client.tunings.get(name=nombre_trabajo)
        print(f"estado actual: {trabajo.state}")
        if trabajo.state in estados_finales:
            return trabajo
        time.sleep(intervalo_segundos)


if __name__ == "__main__":
    print(f"{len(examples)} ejemplos construidos")
    print("Primer ejemplo:", examples[0])

    print("\n--- Balance por categoría ---")
    balance = chequear_balance(ejemplos_raw, "categoria")
    for categoria, cuenta in balance.most_common():
        print(f"{categoria}: {cuenta}")
    razon = max(balance.values()) / min(balance.values())
    print(f"razon max/min: {razon:.1f}x")

    print("\n--- Validación del dataset limpio ---")
    problemas_reales = validar_dataset(examples, claves_esperadas=CLAVES_ESPERADAS_HORECA)
    print(f"{len(examples)} ejemplos, {len(problemas_reales)} problemas encontrados")
    for problema in problemas_reales:
        print(" -", problema)

    print("\n--- Prueba negativa (validador detectando ejemplos rotos) ---")
    examples_rotos = examples + [
        types.TuningExample(text_input="", output='{"categoria": "pedido_compra"}'),
        types.TuningExample(text_input=ejemplos_raw[0][0], output="no es json"),
    ]
    problemas_rotos = validar_dataset(examples_rotos, claves_esperadas=CLAVES_ESPERADAS_HORECA)
    print(f"{len(examples_rotos)} ejemplos, {len(problemas_rotos)} problemas encontrados")
    for problema in problemas_rotos:
        print(" -", problema)
