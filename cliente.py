"""
Modulo 2 - Bloques 12, 13 y 14: cliente.py
Cliente modular de consumo de APIs con clasificación de errores,
reintentos con backoff exponencial, jitter y registro de costos acumulados.
"""

import os
import time
import random
import requests
from dotenv import load_dotenv
from typing import Callable, Any

from cliente_api import armar_peticion, extraer_texto

# Cargar variables del entorno
load_dotenv()

# Configuración de credenciales y URL oficial del modelo
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.8-flash:generateContent?key={GEMINI_API_KEY}"
)
headers = {"Content-Type": "application/json"}

# Constantes para auditoría de costos de tokens
COSTO_POR_1000_TOKENS = 0.002  # Valor referencial según tarifas oficiales
costo_total_sesion = 0.0


def enviar_peticion(cuerpo: dict) -> dict:
    """
    Realiza la llamada HTTP a la API y clasifica la respuesta:
    - 401 / 403: Credencial inválida -> lanza ValueError (no se reintenta).
    - 429 / 5xx / caídas de red: Límite de uso o caída momentánea -> lanza TimeoutError (sí se reintenta).
    - 200: Respuesta exitosa -> retorna el JSON.
    """
    try:
        respuesta = requests.post(URL, json=cuerpo, headers=headers, timeout=10)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        raise TimeoutError("No se pudo contactar al proveedor. Puede ser momentáneo.")

    if respuesta.status_code in (401, 403):
        raise ValueError("Credencial inválida o sin permiso. Revisa tu GEMINI_API_KEY.")

    if respuesta.status_code == 429 or respuesta.status_code >= 500:
        raise TimeoutError("El proveedor no pudo procesar la petición. Hay que esperar y reintentar.")

    respuesta.raise_for_status()
    return respuesta.json()


def con_reintento(
    funcion: Callable[[], Any],
    intentos: int = 3,
    espera_base: float = 2.0,
    espera_maxima: float = 30.0,
) -> Any:
    """
    Ejecuta una función y reintenta si lanza TimeoutError aplicando:
    1. Backoff exponencial: duplica el tiempo de espera en cada intento.
    2. Techo máximo: nunca supera 'espera_maxima'.
    3. Jitter: aleatoriedad para evitar colisiones entre peticiones simultáneas.
    Si se agotan los intentos, propaga el TimeoutError original.
    """
    for intento in range(1, intentos + 1):
        try:
            return funcion()
        except TimeoutError:
            if intento == intentos:
                raise
            espera = min(espera_base * (2 ** (intento - 1)), espera_maxima)
            espera_con_jitter = espera + random.uniform(0, espera * 0.1)
            print(f"Intento {intento} de {intentos} falló por límite de uso. Reintentando en {espera_con_jitter:.1f}s...")
            time.sleep(espera_con_jitter)


def registrar_costo(respuesta: dict) -> float:
    """
    Calcula el costo estimado de la operación basado en los tokens consumidos
    y lo acumula en la variable de sesión 'costo_total_sesion'.
    """
    global costo_total_sesion
    uso = respuesta.get("usageMetadata", {})
    total_tokens = uso.get("totalTokenCount", 0)
    costo_estimado = (total_tokens / 1000) * COSTO_POR_1000_TOKENS
    costo_total_sesion += costo_estimado
    print(f"Operación: {total_tokens} tokens, costo estimado: ${costo_estimado:.5f} (acumulado: ${costo_total_sesion:.5f})")
    return costo_estimado


def consultar_modelo(pregunta: str, system_instruction: str = "") -> str:
    """
    Punto de entrada principal del cliente modular:
    1. Prepara el cuerpo de la petición.
    2. Envía la llamada protegida por reintentos automáticos.
    3. Si se agotan los reintentos, traduce el TimeoutError en RuntimeError amigable.
    4. Registra el costo de la operación si fue exitosa.
    5. Devuelve el texto limpio generado por el modelo.
    """
    cuerpo = armar_peticion(pregunta, system_instruction)
    try:
        respuesta = con_reintento(lambda: enviar_peticion(cuerpo))
    except TimeoutError:
        raise RuntimeError("El servicio no respondió tras varios intentos. Intenta más tarde.")

    registrar_costo(respuesta)
    return extraer_texto(respuesta)


if __name__ == "__main__":
    print("--- Probando consulta normal ---")
    try:
        respuesta = consultar_modelo("¿Cuál es la capital de Chile?")
        print(f"Respuesta: {respuesta.strip()}")
        print(f"Costo acumulado de la sesión: ${costo_total_sesion:.5f}")
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
