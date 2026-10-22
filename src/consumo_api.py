import os
import re
import json
from typing import Tuple, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Precios de referencia por token (Gemini 2.5 Flash / 1.5 Flash estándar)
# Entrada: $0.075 por 1M tokens ($0.000000075 por token)
# Salida:  $0.30  por 1M tokens ($0.000000300 por token)
PRECIO_TOKEN_ENTRADA_DEFAULT = 0.000000075
PRECIO_TOKEN_SALIDA_DEFAULT = 0.000000300


def armar_peticion(
    pregunta_usuario: str,
    system_instruction: str,
    response_schema: Optional[Dict[str, Any]] = None,
    modelo: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Construye la URL y el cuerpo JSON para invocar la API REST de Google Gemini.
    Valida preventivamente las entradas y soporta responseSchema estructurado.
    """
    if not pregunta_usuario or not pregunta_usuario.strip():
        raise ValueError("La entrada del usuario no puede estar vacía o contener solo espacios.")
    
    if not system_instruction or not system_instruction.strip():
        raise ValueError("La instrucción de sistema no puede estar vacía.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró GEMINI_API_KEY en las variables de entorno.")

    modelo_final = modelo or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_final}:generateContent?key={api_key}"

    body: Dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": system_instruction.strip()}]
        },
        "contents": [
            {"parts": [{"text": pregunta_usuario.strip()}]}
        ]
    }

    if response_schema is not None:
        body["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }

    return url, body


def enviar_peticion(url: str, body: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """
    Envía la petición HTTP POST al endpoint del modelo con control de errores
    de conexión, respuestas HTTP de error (401, 429) y validación de formato JSON.
    """
    try:
        respuesta = requests.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
    except requests.exceptions.ConnectionError as err:
        raise RuntimeError(f"Error de red: No se pudo conectar con el servicio de IA. Revisa tu conexión. Detalle: {err}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Tiempo de espera agotado ({timeout}s) al comunicarse con el proveedor del modelo.")

    # Manejo específico de códigos de estado HTTP
    if respuesta.status_code == 401:
        raise PermissionError("Error de autenticación (401): La API Key de Gemini es inválida o expiró.")
    elif respuesta.status_code == 429:
        raise RuntimeError("Cuota excedida (429): Límite de solicitudes alcanzado. Espera un momento antes de reintentar.")
    elif respuesta.status_code >= 400:
        detalle = respuesta.text
        raise RuntimeError(f"Error HTTP {respuesta.status_code} devuelto por el servicio de IA: {detalle}")

    try:
        return respuesta.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError("La respuesta devuelta por el servidor no contiene un JSON válido.")


def extraer_texto(datos: Dict[str, Any]) -> str:
    """
    Navega la estructura jerárquica de la respuesta de Gemini para extraer el texto generado.
    """
    try:
        return datos["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("La respuesta del modelo no contiene texto en la ruta esperada (candidates[0].content.parts[0].text).")


def parsear_respuesta_json(texto: str) -> Dict[str, Any]:
    """
    Limpia bloques de markdown (```json ... ```) y valida que el texto sea un objeto JSON válido.
    """
    texto_limpio = texto.strip()
    texto_limpio = re.sub(r"^```(?:json)?\s*", "", texto_limpio)
    texto_limpio = re.sub(r"\s*```$", "", texto_limpio)

    try:
        datos = json.loads(texto_limpio)
    except json.JSONDecodeError as err:
        raise ValueError(f"La respuesta generada no pudo interpretarse como JSON: {err}")

    if not isinstance(datos, dict):
        raise ValueError("El JSON generado debe ser un objeto/diccionario, no una lista o tipo primitivo.")

    return datos


def calcular_costo(
    datos: Dict[str, Any],
    precio_entrada: float = PRECIO_TOKEN_ENTRADA_DEFAULT,
    precio_salida: float = PRECIO_TOKEN_SALIDA_DEFAULT
) -> Dict[str, Any]:
    """
    Lee 'usageMetadata' de la respuesta del modelo y calcula el costo exacto en USD.
    Retorna métricas de tokens y costo desglosado.
    """
    uso = datos.get("usageMetadata", {})
    tokens_entrada = uso.get("promptTokenCount", 0)
    tokens_salida = uso.get("candidatesTokenCount", 0)
    tokens_total = uso.get("totalTokenCount", tokens_entrada + tokens_salida)

    costo_entrada = tokens_entrada * precio_entrada
    costo_salida = tokens_salida * precio_salida
    costo_total = costo_entrada + costo_salida

    return {
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_total": tokens_total,
        "costo_entrada_usd": round(costo_entrada, 8),
        "costo_salida_usd": round(costo_salida, 8),
        "costo_total_usd": round(costo_total, 8)
    }


def ejecutar_flujo_completo(
    pregunta_usuario: str,
    system_instruction: str,
    response_schema: Optional[Dict[str, Any]] = None,
    modelo: Optional[str] = None
) -> Dict[str, Any]:
    """
    Función orquestadora de alto nivel que coordina todas las etapas modulares:
    armado -> envío -> extracción de texto -> parseo JSON -> cálculo de costo.
    """
    url, body = armar_peticion(pregunta_usuario, system_instruction, response_schema, modelo)
    respuesta_cruda = enviar_peticion(url, body)
    texto_generado = extraer_texto(respuesta_cruda)
    datos_estructurados = parsear_respuesta_json(texto_generado)
    metricas_costo = calcular_costo(respuesta_cruda)

    return {
        "status": "success",
        "datos": datos_estructurados,
        "costo_metricas": metricas_costo
    }
