"""
cliente_api.py - Comunicación base con la API de Gemini (Módulo 2).

Contiene las funciones elementales para preparar la petición, enviarla
al proveedor clasificando los códigos de estado HTTP, y procesar la respuesta.
"""

import os
import requests
from dotenv import load_dotenv

# Cargamos las variables de entorno desde el archivo .env
load_dotenv()

# Configuración básica de conexión
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.8-flash:generateContent?key={GEMINI_API_KEY}"
)
headers = {"Content-Type": "application/json"}


def armar_peticion(pregunta: str, system_instruction: str = "") -> dict:
    """
    Construye el cuerpo en formato JSON que espera la API de Gemini.
    Permite incluir una instrucción de sistema si se especifica.
    """
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La pregunta debe ser una cadena de texto no vacía.")

    cuerpo = {
        "contents": [
            {
                "parts": [
                    {"text": pregunta}
                ]
            }
        ]
    }

    if system_instruction:
        cuerpo["systemInstruction"] = {
            "parts": [
                {"text": system_instruction}
            ]
        }

    return cuerpo


def enviar_peticion(cuerpo: dict) -> dict:
    """
    Envía la petición HTTP POST a la API y traduce las respuestas en excepciones:
    - 401 o 403: Credencial inválida -> lanza ValueError (no reintentable).
    - 429 o >= 500: Límite de uso o caída momentánea -> lanza TimeoutError (reintentable).
    - 200: Devuelve el diccionario JSON con la respuesta del modelo.
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


def extraer_texto(respuesta: dict) -> str:
    """Extrae el contenido de texto generado dentro del JSON de la respuesta."""
    try:
        return respuesta["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("La respuesta de la API no contiene el formato esperado.")
