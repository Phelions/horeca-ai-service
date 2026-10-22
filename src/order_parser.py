"""
================================================================================
MÓDULO: order_parser.py
PROPÓSITO: Extracción y normalización de pedidos gastronómicos informales.
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Toma un mensaje escrito de forma libre o informal (como un mensaje de WhatsApp
o la transcripción de un audio que manda un cocinero: "Carlos, anota 2 cajas de
tomate y 3 sacos de harina") y lo convierte en una lista de compras limpia,
ordenada y 100% estructurada en formato JSON.

¿CON QUIÉN SE COMUNICA?
-----------------------
1. Con el archivo local `.env`: Lee la clave secreta `GEMINI_API_KEY` para no
   dejar contraseñas escritas en el código.
2. Con la nube de Google (Google Gemini API): Envía el texto informal y recibe
   la respuesta generada por el modelo de Inteligencia Artificial (Gemini 3.8 Flash).

¿CÓMO SE COMUNICA?
------------------
Utiliza el SDK oficial `google-genai`. Le pasa a Gemini una "instrucción de sistema"
(para que actúe como un comprador profesional de restaurante) y le exige
que la respuesta cumpla obligatoriamente con el esquema definido con Pydantic
(`PedidoExtraido`). Así evitamos que el modelo responda con texto conversacional
o párrafos sueltos.
"""

import os
import json
import re
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Cargar las credenciales y variables guardadas en el archivo .env
load_dotenv()

# ==============================================================================
# PRECIOS OFICIALES DE GEMINI 3.8 FLASH (Octubre 2026)
# ==============================================================================
# El modelo cobra por cada 1.000 tokens que lee (entrada) y genera (salida):
# - Entrada: $0.00075 USD por cada 1.000 tokens ($0.00000075 por token)
# - Salida:  $0.00375 USD por cada 1.000 tokens ($0.00000375 por token)
PRECIO_TOKEN_ENTRADA_USD = 0.00000075
PRECIO_TOKEN_SALIDA_USD = 0.00000375


# ==============================================================================
# 1. ESQUEMAS DE DATOS (PYDANTIC)
# ==============================================================================
# Estos esquemas actúan como un "molde" o contrato estricto: definen qué campos
# debe tener cada producto y qué tipo de dato se espera (texto, número, etc.).

class ItemPedido(BaseModel):
    """
    Representa un solo producto dentro del pedido de cocina.
    
    Campos:
    - producto_mencionado: Nombre tal como lo dijo el cocinero (ej: 'tomate limachino').
    - cantidad: Número exacto pedido. Si dice 'media caja', se guarda como 0.5.
    - unidad_mencionada: Unidad de medida (ej: 'caja', 'kilo', 'saco', 'bidon').
    - observaciones: Detalles opcionales de maduración, estado o marca (ej: 'bien maduro').
    """
    producto_mencionado: str = Field(
        description="Nombre o descripción informal del producto solicitado (ej: tomate limachino, posta negra, harina)."
    )
    cantidad: float = Field(
        description="Cantidad numérica solicitada. Si se menciona media unidad, registrar como 0.5."
    )
    unidad_mencionada: str = Field(
        description="Unidad de medida señalada (ej: caja, saco, kilo, bidon, pieza, bandeja, bolsa, unidad)."
    )
    observaciones: Optional[str] = Field(
        default=None,
        description="Notas adicionales como estado, maduración, marca o corte específico si se menciona."
    )


class PedidoExtraido(BaseModel):
    """
    Representa la orden completa de compra, que contiene la lista de ítems extraídos.
    """
    items: List[ItemPedido] = Field(
        description="Lista exhaustiva de todos los insumos solicitados en el mensaje."
    )


# ==============================================================================
# 2. CLASE PRINCIPAL: OrderParser
# ==============================================================================

class OrderParser:
    """
    Clase que orquesta todo el proceso:
    1. Valida la entrada antes de gastar recursos.
    2. Conecta con Gemini y le envía el pedido informal.
    3. Asegura que la salida sea un JSON válido bajo el esquema PedidoExtraido.
    4. Calcula el costo en dólares y los tokens consumidos en vivo.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Inicializa el cliente de Gemini verificando que la API key exista en el entorno.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró GEMINI_API_KEY en las variables de entorno. "
                "Verifica que tu archivo .env exista y contenga una clave válida."
            )
        
        # Modelo oficial recomendado (Gemini 3.8 Flash para velocidad y agentes)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        
        # Cliente oficial del SDK de Google GenAI
        self.client = genai.Client(api_key=api_key)
        
        # Instrucción de Sistema: Fija la 'personalidad' y reglas de negocio del modelo
        self.system_instruction = (
            "Eres un asistente de adquisiciones para el sector gastronómico (HORECA). "
            "Tu misión es analizar mensajes informales o transcripciones de audio enviadas por "
            "cocineros o administradores y extraer con precisión quirúrgica cada producto, "
            "su cantidad numérica y la unidad de medida señalada. "
            "Si un producto no especifica unidad clara pero sí cantidad, infiere la unidad más "
            "lógica o usa 'unidad'. Extrae únicamente los insumos reales que constituyen una compra."
        )

    def validar_entrada(self, texto_pedido: str) -> None:
        """
        VALIDACIÓN PREVENTIVA (Defensa antes de llamar a la API):
        Si alguien pasa un texto vacío, solo espacios o un tipo de dato incorrecto,
        detiene la ejecución inmediatamente con un error claro. Esto ahorra dinero
        y evita enviar llamadas inútiles a Google.
        """
        if not isinstance(texto_pedido, str):
            raise ValueError(f"La entrada debe ser texto (str), pero se recibió: {type(texto_pedido).__name__}")
        if not texto_pedido or not texto_pedido.strip():
            raise ValueError("El texto del pedido no puede estar vacío ni contener solo espacios.")

    def calcular_costo_tokens(self, prompt_tokens: int, candidates_tokens: int) -> Dict[str, Any]:
        """
        CÁLCULO FINANCIERO EN VIVO:
        Calcula exactamente cuánto dinero costó esta consulta específica, multiplicando
        los tokens de entrada y salida por sus tarifas oficiales en USD.
        """
        costo_in = prompt_tokens * PRECIO_TOKEN_ENTRADA_USD
        costo_out = candidates_tokens * PRECIO_TOKEN_SALIDA_USD
        costo_total = costo_in + costo_out
        
        return {
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "total_tokens": prompt_tokens + candidates_tokens,
            "costo_entrada_usd": round(costo_in, 8),
            "costo_salida_usd": round(costo_out, 8),
            "costo_total_usd": round(costo_total, 8)
        }

    def parsear_texto(self, texto_pedido: str) -> Dict[str, Any]:
        """
        MÉTODO PRINCIPAL:
        Transforma el mensaje informal en la estructura JSON validada.
        
        Flujo de ejecución:
        1. Valida el texto.
        2. Configura Gemini con baja temperatura (0.1) para que sea consistente y no invente.
        3. Realiza la llamada remota forzando salida JSON (`response_schema=PedidoExtraido`).
        4. Captura posibles errores de conexión o límites de uso sin botar la aplicación.
        5. Devuelve un diccionario con el pedido, métricas de tokens y costos.
        """
        # Paso 1: Validación temprana
        self.validar_entrada(texto_pedido)

        try:
            # Paso 2: Configuración de la llamada
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                response_mime_type="application/json",
                response_schema=PedidoExtraido,
                temperature=0.1,  # Temperatura baja = Respuestas predecibles y precisas
            )

            # Paso 3: Comunicación con la API de Google Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Extrae los insumos del siguiente pedido:\n\n\"{texto_pedido.strip()}\"",
                config=config,
            )

            # Paso 4: Extracción de tokens y auditoría de costo
            usage = response.usage_metadata
            p_tokens = getattr(usage, "prompt_token_count", 0) or 0
            c_tokens = getattr(usage, "candidates_token_count", 0) or 0
            metricas_costo = self.calcular_costo_tokens(p_tokens, c_tokens)

            # Paso 5: Interpretación del JSON devuelto
            if response.parsed:
                # Si el SDK ya convirtió la respuesta a nuestro objeto Pydantic
                datos_pedido = response.parsed.model_dump()
            else:
                # Si vino como texto plano, limpiamos posibles marcas de markdown
                texto_limpio = response.text.strip()
                texto_limpio = re.sub(r"^```(?:json)?\s*", "", texto_limpio)
                texto_limpio = re.sub(r"\s*```$", "", texto_limpio)
                datos_pedido = json.loads(texto_limpio)

            # Validamos con Pydantic para garantizar conformidad total
            pedido_validado = PedidoExtraido(**datos_pedido).model_dump()

            return {
                "status": "success",
                "pedido": pedido_validado,
                "token_metrics": metricas_costo,
                "model_used": self.model_name
            }

        except APIError as api_err:
            # Captura errores que devuelve Google (ej: clave incorrecta, cuota saturada)
            return {
                "status": "error",
                "error_type": "Gemini_API_Error",
                "message": f"Error devuelto por la API del proveedor: {api_err}"
            }
        except json.JSONDecodeError as json_err:
            # Captura si el texto devuelto no tuvo un formato JSON válido
            return {
                "status": "error",
                "error_type": "JSON_Decode_Error",
                "message": f"No se pudo decodificar la salida como JSON válido: {json_err}"
            }
        except Exception as e:
            # Captura cualquier otra falla inesperada sin que el programa termine abruptly
            return {
                "status": "error",
                "error_type": "Unexpected_Error",
                "message": f"Falla inesperada durante el procesamiento: {e}"
            }


# ==============================================================================
# 3. EJECUCIÓN DIRECTA (DEMOSTRACIÓN LOCAL)
# ==============================================================================
# Si ejecutas 'python src/order_parser.py' directamente en tu consola,
# este bloque se ejecuta para que veas el resultado con un ejemplo real.

if __name__ == "__main__":
    mensaje_ejemplo = (
        "Hola don Carlos, para mañana necesitamos con urgencia 2 cajas de tomate limachino "
        "bien firme, 3 sacos de harina panadera, unos 10 kilos de posta negra limpia "
        "y 2 bidones de lavaloza concentrado. Ah y si tiene palta hass, mándeme 1 caja. Gracias!"
    )

    print("=" * 70)
    print(" INICIANDO EXTRACCIÓN DE PEDIDO HORECA CON GEMINI 3.8 FLASH")
    print("=" * 70)
    print(f"Mensaje informal recibido de la cocina:\n\"{mensaje_ejemplo}\"\n")

    parser = OrderParser()
    resultado = parser.parsear_texto(mensaje_ejemplo)

    print("Resultado estructurado (JSON) y métricas de consumo:")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))