"""
cliente.py - Cliente modular resiliente para HORECA AI Service (Módulo 2).

Integra clasificación de errores HTTP (401 vs 429/5xx), reintentos automáticos
con backoff exponencial, techo máximo y jitter aleatorio, auditoría de tokens
y una interfaz simple (consultar_modelo) pensada para resistir ráfagas de pedidos
nocturnos en restaurantes sin saturar cuotas ni interrumpir la operación.
"""

import os
import time
import random
import requests
from dotenv import load_dotenv
from typing import Callable, Any

from cliente_api import armar_peticion, extraer_texto, SYSTEM_INSTRUCTION_HORECA

# Cargar variables de entorno del archivo .env
load_dotenv()

# Configuración de credenciales y URL del modelo Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODELO_ACTUAL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODELO_ACTUAL}:generateContent?key={GEMINI_API_KEY}"
)
headers = {"Content-Type": "application/json"}

# Constantes de auditoría de costos de tokens (referencial oficial Gemini 3.8 Flash)
COSTO_POR_1000_TOKENS = 0.002
costo_total_sesion = 0.0


def enviar_peticion(cuerpo: dict) -> dict:
    """
    Realiza la llamada HTTP a la API y clasifica la respuesta según el tipo de falla:
    - 401 / 403: Credencial inválida -> lanza ValueError (falla de configuración, no se reintenta).
    - 429 / 5xx / caídas de red: Límite de cuota o indisponibilidad momentánea -> lanza TimeoutError (sí se reintenta).
    - 200: Respuesta exitosa -> retorna el diccionario JSON.
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
    Envuelve cualquier función protegida ante fallas momentáneas (TimeoutError) aplicando:
    1. Backoff exponencial: duplica el tiempo de espera en cada intento sucesivo.
    2. Techo máximo: nunca supera 'espera_maxima' segundos.
    3. Jitter aleatorio: agrega hasta un 10% de variación para evitar que múltiples
       pedidos de cocina concurrentes colisionen contra el mismo límite de uso.
    Si la función lanza ValueError (error de credenciales), NO reintenta y deja propagar el error.
    Si se agotan los reintentos permitidos, vuelve a lanzar el TimeoutError original.
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
    Lee los tokens reales consumidos desde 'usageMetadata' en la respuesta del proveedor,
    calcula el costo estimado en dólares y lo acumula en 'costo_total_sesion'.
    Solo se ejecuta tras una respuesta HTTP exitosa para no contar transacciones fallidas.
    """
    global costo_total_sesion
    uso = respuesta.get("usageMetadata", {})
    total_tokens = uso.get("totalTokenCount", 0)
    costo_estimado = (total_tokens / 1000) * COSTO_POR_1000_TOKENS
    costo_total_sesion += costo_estimado
    print(f"Operación: {total_tokens} tokens, costo estimado: ${costo_estimado:.5f} (acumulado: ${costo_total_sesion:.5f})")
    return costo_estimado


def consultar_modelo(
    pregunta: str,
    system_instruction: str = "",
) -> str:
    """
    Punto de entrada modular para la extracción o consulta al modelo en HORECA AI Service:
    1. Prepara la petición separando el mensaje del pedido y la directriz de adquisiciones.
    2. Ejecuta la petición protegida bajo el mecanismo automático de 'con_reintento'.
    3. Si se agotan los reintentos, captura el TimeoutError y lanza RuntimeError amigable
       para el consumidor del servicio.
    4. Si la respuesta fue exitosa, audita el consumo de tokens y acumula el costo en dólares.
    5. Retorna el texto limpio listo para usar.
    """
    instruccion = system_instruction if system_instruction else SYSTEM_INSTRUCTION_HORECA
    cuerpo = armar_peticion(pregunta, instruccion)

    try:
        respuesta = con_reintento(lambda: enviar_peticion(cuerpo))
    except TimeoutError:
        raise RuntimeError("El servicio no respondió tras varios intentos. Intenta más tarde.")

    registrar_costo(respuesta)
    return extraer_texto(respuesta)


if __name__ == "__main__":
    print("=" * 70)
    print(" 🍳 HORECA AI SERVICE - CLIENTE MODULAR CON RESILIENCIA Y COSTO")
    print("=" * 70)
    
    # Ejemplo de consulta real de abastecimiento gastronómico
    pedido_prueba = (
        "Hola Carlos, para mañana a primera hora necesitamos 2 cajas de tomate limachino "
        "bien firme, 3 sacos de harina panadera y 10 kilos de posta negra limpia. "
        "¿Podrías confirmarme disponibilidad y tiempos estimados de despacho?"
    )

    print("\n[CONSULTA GASTRONÓMICA ENVIADA]:")
    print(f"\"{pedido_prueba}\"\n")

    try:
        resultado = consultar_modelo(pedido_prueba)
        print("[RESPUESTA DEL ASISTENTE HORECA]:")
        print(resultado.strip())
        print("-" * 70)
        print(f"💰 Costo acumulado de la sesión: ${costo_total_sesion:.5f} USD")
    except ValueError as err_cfg:
        print(f"❌ Error de configuración o credenciales: {err_cfg}")
    except RuntimeError as err_red:
        print(f"⚠️ El servicio no pudo completar el pedido: {err_red}")
