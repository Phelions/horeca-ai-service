"""
verificar_entorno.py - Verificación del entorno de personalización y tipos del SDK (Bloque 29 / U2.6).

Comprueba sin consumir cuota ni lanzar entrenamientos:
1. Versión instalada de google-genai y presencia de GEMINI_API_KEY en .env.
2. Disponibilidad de TuningMode, AdapterSize y campos de CreateTuningJobConfig.
3. Clasificación descriptiva de errores de autenticación, cuota y conexión (explicar_error).
"""

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.genai import types


PAQUETES_GATE2: List[str] = [
    "langchain-core",
    "langchain-chroma",
    "langgraph",
    "python-dotenv",
    "langchain-google-genai",
    "google-genai",
]
VARIABLES_GATE2: List[str] = ["GEMINI_API_KEY"]
INDICE_GATE2_DIR: str = "vectorstore"


def explicar_error(error: Exception) -> str:
    """Traduce excepciones comunes de la API de Gemini a mensajes accionables."""
    mensaje = str(error)
    if "API key" in mensaje or "401" in mensaje or "403" in mensaje:
        return "Problema de autenticación: revisa GEMINI_API_KEY en tu archivo .env."
    if "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
        return "Cuota o límite de peticiones agotado: espera unos minutos y reintenta."
    if "Connection" in mensaje or "timed out" in mensaje:
        return "Problema de conexión: revisa tu internet y vuelve a intentarlo."
    return "Error inesperado: " + mensaje


def verificar_entorno_tuning() -> Dict[str, object]:
    """Verifica google-genai, la presencia de GEMINI_API_KEY y los tipos de tuning del SDK."""
    load_dotenv(override=True)
    try:
        version_sdk = version("google-genai")
    except PackageNotFoundError as error:
        raise SystemExit(
            "Falta la librería. Ejecuta: pip install google-genai python-dotenv"
        ) from error

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit('Falta GEMINI_API_KEY. Agrégala al archivo ".env" de tu proyecto.')

    modos = [str(modo) for modo in types.TuningMode]
    tamanios_adapter = [str(tamanio) for tamanio in types.AdapterSize]
    campos_config: Dict[str, str] = {}
    for campo in ("epoch_count", "learning_rate_multiplier", "batch_size"):
        if campo in types.CreateTuningJobConfig.model_fields:
            campos_config[campo] = types.CreateTuningJobConfig.model_fields[campo].description or ""
        else:
            campos_config[campo] = "este campo no existe en tu versión del SDK"

    return {
        "version_sdk": version_sdk,
        "api_key_detectada": True,
        "modos_tuning": modos,
        "tamanios_adapter": tamanios_adapter,
        "campos_config": campos_config,
    }


def imprimir_resumen_entorno(resumen: Dict[str, object]) -> None:
    """Imprime el resultado legible de verificación del entorno y tipos del SDK."""
    print("google-genai instalado, versión", resumen["version_sdk"])
    print("GEMINI_API_KEY detectada")
    print("\nModos de tuning disponibles:")
    for modo in resumen["modos_tuning"]:  # type: ignore[attr-defined]
        print(" ", modo)
    print("\nTamanios de adapter disponibles (solo aplica a PEFT):")
    for tamanio in resumen["tamanios_adapter"]:  # type: ignore[attr-defined]
        print(" ", tamanio)
    print("\nCampos de CreateTuningJobConfig:")
    for campo, desc in resumen["campos_config"].items():  # type: ignore[attr-defined]
        print(f" {campo} -> {desc}")


def verificar_entorno_gate2(
    paquetes: List[str] = PAQUETES_GATE2,
    variables: List[str] = VARIABLES_GATE2,
    indice_dir: str = INDICE_GATE2_DIR,
) -> Dict[str, Any]:
    """Comprueba Python, dependencias, variables (.env sin imprimir clave) e índice persistido."""
    load_dotenv()
    problemas: List[str] = []
    versiones: Dict[str, str] = {}
    estado_variables: Dict[str, str] = {}

    if sys.version_info < (3, 10):
        problemas.append("usar Python 3.10 o superior")

    for paquete in paquetes:
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            versiones[paquete] = "NO instalado"
            problemas.append(f"instalar {paquete}")

    for nombre in variables:
        definida = bool(os.getenv(nombre))
        estado_variables[nombre] = "definida" if definida else "FALTA"
        if not definida:
            problemas.append(f"definir {nombre} en el archivo .env")

    indice_ok = os.path.isdir(indice_dir) and bool(os.listdir(indice_dir))
    if not indice_ok:
        problemas.append(f"indexar el corpus en {indice_dir}/")

    return {
        "python": sys.version.split()[0],
        "versiones": versiones,
        "variables": estado_variables,
        "indice_dir": indice_dir,
        "indice_ok": indice_ok,
        "problemas": problemas,
        "listo": not problemas,
    }


def imprimir_resumen_gate2(estado: Dict[str, Any]) -> None:
    """Imprime la verificación completa del entorno Gate 2 sin exponer claves."""
    print("Python:", estado["python"])
    for paquete, ver in estado["versiones"].items():
        print(f"{paquete}: {ver}")
    for nombre, flag in estado["variables"].items():
        print(f"{nombre}: {flag}")
    indice_dir = estado["indice_dir"]
    print(f"indice en {indice_dir}/: {'ok' if estado['indice_ok'] else 'NO existe o esta vacio'}")
    print()
    if estado["problemas"]:
        print("Pendiente: " + "; ".join(estado["problemas"]))
    else:
        print("Entorno listo para la corrida final")


if __name__ == "__main__":
    solo_gate2 = "--gate2" in sys.argv
    estado_gate2 = verificar_entorno_gate2()
    imprimir_resumen_gate2(estado_gate2)
    if not estado_gate2["listo"]:
        sys.exit(1)
    if not solo_gate2:
        print()
        info = verificar_entorno_tuning()
        imprimir_resumen_entorno(info)
