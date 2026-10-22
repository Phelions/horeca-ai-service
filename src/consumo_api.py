"""
================================================================================
MÓDULO: consumo_api.py
PROPÓSITO: Cliente modular desacoplado para invocar la API REST de Gemini.
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Contiene las funciones básicas y atómicas (con una única responsabilidad cada una)
para comunicarse con la API de Inteligencia Artificial mediante peticiones HTTP directas.
Cumple con la arquitectura de 5 funciones exigida en los módulos U1.3, U1.4 y U1.5:
1. armar_peticion(): Prepara la URL y el cuerpo del mensaje en formato JSON.
2. enviar_peticion(): Hace el envío por internet y revisa si la red o la clave fallan.
3. extraer_texto(): Saca el texto de la respuesta ignorando la metadata técnica.
4. parsear_respuesta_json(): Limpia marcas de formato markdown y crea el diccionario.
5. calcular_costo(): Calcula el costo en dólares reales según los tokens utilizados.

¿CON QUIÉN SE COMUNICA?
-----------------------
- Con el endpoint REST oficial de Google Gemini (`generativelanguage.googleapis.com`)
  a través del protocolo HTTPS estándar usando la librería `requests`.
- Con el archivo `.env` para obtener la clave secreta de forma protegida.

¿CÓMO SE COMUNICA?
------------------
Envía un paquete `POST` con encabezado `application/json`, llevando la instrucción
de sistema en un canal (`systemInstruction`) y la pregunta en otro (`contents`),
garantizando además la restricción de esquema con `responseSchema`.
"""

import os
import re
import json
from typing import Tuple, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ==============================================================================
# PRECIOS OFICIALES DE REFERENCIA (Gemini 3.8 Flash - Octubre 2026)
# ==============================================================================
# Entrada: $0.00075 por 1k tokens ($0.00000075 por token)
# Salida:  $0.00375 por 1k tokens ($0.00000375 por token)
PRECIO_TOKEN_ENTRADA_DEFAULT = 0.00000075
PRECIO_TOKEN_SALIDA_DEFAULT = 0.00000375


def armar_peticion(
    pregunta_usuario: str,
    system_instruction: str,
    response_schema: Optional[Dict[str, Any]] = None,
    modelo: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    PASO 1: ARMAR LA PETICIÓN
    
    ¿Qué hace?
    - Valida que no envíes un mensaje en blanco (para no gastar dinero en balde).
    - Lee tu API Key desde el archivo `.env`.
    - Arma la dirección URL de Google a la que hay que llamar.
    - Construye el cuerpo (Body) en formato JSON, separando la regla de sistema
      de lo que escribió el usuario. Si se entrega un 'response_schema', le exige
      a Google que la respuesta sea un JSON con esa forma exacta.
      
    ¿Qué devuelve?
    - Una tupla con dos cosas: (url_del_servicio, cuerpo_json_a_enviar).
    """
    if not pregunta_usuario or not pregunta_usuario.strip():
        raise ValueError("La entrada del usuario no puede estar vacía o contener solo espacios.")
    
    if not system_instruction or not system_instruction.strip():
        raise ValueError("La instrucción de sistema no puede estar vacía.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró GEMINI_API_KEY en las variables de entorno.")

    modelo_final = modelo or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_final}:generateContent?key={api_key}"

    body: Dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": system_instruction.strip()}]
        },
        "contents": [
            {"parts": [{"text": pregunta_usuario.strip()}]}
        ]
    }

    # Si se pide forzar estructura JSON, se añade generationConfig
    if response_schema is not None:
        body["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }

    return url, body


def enviar_peticion(url: str, body: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """
    PASO 2: ENVIAR LA PETICIÓN POR INTERNET
    
    ¿Qué hace?
    - Hace la llamada HTTP POST real hacia los servidores de Google.
    - Establece un tiempo de espera máximo (timeout = 30 segundos) para que tu
      programa no se quede congelado si la red anda lenta.
    - Si se cae el WiFi o no hay internet, captura el error de inmediato y te da
      un mensaje comprensible en vez de botar el programa.
    - Revisa códigos de error de Google:
      * 401: Tu clave de API es incorrecta o venció.
      * 429: Hiciste demasiadas peticiones muy rápido (límite de cuota alcanzado).
      * 400 o superior: Algún dato de la petición no fue aceptado.
      
    ¿Qué devuelve?
    - La respuesta cruda de Google convertida a diccionario de Python.
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

    # Manejo de respuestas de error específicas de la API
    if respuesta.status_code == 401:
        raise PermissionError("Error de autenticación (401): La API Key de Gemini es inválida o expiró.")
    elif respuesta.status_code == 429:
        raise RuntimeError("Cuota excedida (429): Límite de solicitudes alcanzado. Espera un momento antes de reintentar.")
    elif respuesta.status_code >= 400:
        detalle = respuesta.text
        raise RuntimeError(f"Error HTTP {respuesta.status_code} devuelto por el servicio de IA: {detalle}")

    # Confirmamos que la respuesta sea JSON válido
    try:
        return respuesta.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError("La respuesta devuelta por el servidor no contiene un JSON válido.")


def extraer_texto(datos: Dict[str, Any]) -> str:
    """
    PASO 3: EXTRAER EL TEXTO GENERADO
    
    ¿Qué hace?
    - La respuesta de Gemini trae muchísima información (tokens, estado, metadatos).
      Esta función navega la jerarquía interna (candidates -> content -> parts -> text)
      y extrae únicamente el texto útil generado por el modelo.
    - Si la respuesta vino vacía o bloqueada por políticas de seguridad, lanza un
      error descriptivo en vez de un confuso 'KeyError'.
    """
    try:
        return datos["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("La respuesta del modelo no contiene texto en la ruta esperada (candidates[0].content.parts[0].text).")


def parsear_respuesta_json(texto: str) -> Dict[str, Any]:
    """
    PASO 4: LIMPIAR Y CONVERTIR A DICCIONARIO
    
    ¿Qué hace?
    - A veces los modelos de lenguaje envuelven el JSON en etiquetas de markdown,
      como '```json ... ```'. Esta función usa expresiones regulares (Regex) para
      eliminar esas marcas y dejar el texto limpio.
    - Luego convierte el texto a un diccionario nativo de Python usando 'json.loads'.
    - Verifica que el resultado sea efectivamente un objeto/diccionario.
    """
    texto_limpio = texto.strip()
    # Eliminar posibles bloques de markdown al inicio y al final
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
    PASO 5: CALCULAR EL COSTO ECONÓMICO DE LA PETICIÓN
    
    ¿Qué hace?
    - Busca dentro de la respuesta el bloque 'usageMetadata' que entrega Gemini.
    - Lee cuántos tokens se enviaron (promptTokenCount) y cuántos se generaron (candidatesTokenCount).
    - Multiplica cada valor por la tarifa oficial correspondiente y los suma.
    
    ¿Qué devuelve?
    - Un resumen completo con la cantidad de tokens y el costo desglosado en dólares (USD).
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
    ORQUESTADOR:
    Llama a las 5 funciones anteriores en el orden exacto:
    1. armar_peticion -> 2. enviar_peticion -> 3. extraer_texto -> 
    4. parsear_respuesta_json -> 5. calcular_costo.
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
