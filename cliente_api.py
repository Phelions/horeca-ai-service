"""
cliente_api.py - Comunicación base con la API de Gemini para HORECA AI Service.

Contiene las funciones atómicas para preparar la petición, enviarla al endpoint
oficial clasificando los códigos de estado HTTP, y extraer el texto de la respuesta.
Diseñado para la ingesta y procesamiento de pedidos gastronómicos informales.
"""

import os
import requests
from dotenv import load_dotenv

# Cargamos las variables de entorno desde el archivo .env
load_dotenv()

# Configuración básica de conexión y modelo
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODELO_ACTUAL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODELO_ACTUAL}:generateContent?key={GEMINI_API_KEY}"
)
headers = {"Content-Type": "application/json"}

# Instrucción de sistema por defecto para el dominio gastronómico HORECA
SYSTEM_INSTRUCTION_HORECA = (
    "Eres un asistente de adquisiciones para el sector gastronómico (HORECA). "
    "Tu misión es analizar mensajes informales o transcripciones de cocina y responder "
    "con claridad sobre insumos, compras mayoristas y abastecimiento de restaurantes."
)


def armar_peticion(pregunta: str, system_instruction: str = "") -> dict:
    """
    Construye el cuerpo en formato JSON que espera la API de Gemini.
    Separa el contenido del usuario de la instrucción de sistema para mantener consistencia.
    """
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La entrada del pedido debe ser una cadena de texto no vacía.")

    cuerpo = {
        "contents": [
            {
                "parts": [
                    {"text": pregunta.strip()}
                ]
            }
        ]
    }

    instruccion_final = system_instruction.strip() if system_instruction else SYSTEM_INSTRUCTION_HORECA
    cuerpo["systemInstruction"] = {
        "parts": [
            {"text": instruccion_final}
        ]
    }

    return cuerpo


def enviar_peticion(cuerpo: dict) -> dict:
    """
    Envía la petición HTTP POST a la API y traduce las respuestas en excepciones:
    - 401 o 403: Credencial inválida -> lanza ValueError (no reintentable).
    - 429 o >= 500: Límite de uso o caída momentánea -> lanza TimeoutError (reintentable).
    - Errores de conexión / timeout de red -> lanza TimeoutError (reintentable).
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
    """Extrae el contenido de texto generado dentro del JSON devuelto por Gemini."""
    try:
        return respuesta["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("La respuesta de la API no contiene el formato esperado.")
