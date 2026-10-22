import os
import json
import re
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Cargar variables de entorno desde .env
load_dotenv()

# Precios oficiales de referencia (Gemini 3.8 Flash - Octubre 2026)
PRECIO_TOKEN_ENTRADA_USD = 0.00000075  # $0.00075 por 1k tokens
PRECIO_TOKEN_SALIDA_USD = 0.00000375   # $0.00375 por 1k tokens


# ==========================================
# 1. Esquemas de Validación (Pydantic)
# ==========================================
class ItemPedido(BaseModel):
    """Representa un ítem individual extraído del pedido informal."""
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
    """Estructura completa de la orden de compra normalizada."""
    items: List[ItemPedido] = Field(
        description="Lista exhaustiva de todos los insumos solicitados en el mensaje."
    )


# ==========================================
# 2. Clase Principal del Parser
# ==========================================
class OrderParser:
    """
    Parser especializado para transformar mensajes informales de cocina (HORECA)
    en órdenes de compra estructuradas y validadas con Pydantic.
    """
    def __init__(self, model_name: Optional[str] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró GEMINI_API_KEY en las variables de entorno. "
                "Verifica tu archivo .env"
            )
        
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=api_key)
        
        self.system_instruction = (
            "Eres un asistente de adquisiciones para el sector gastronómico (HORECA). "
            "Tu misión es analizar mensajes informales o transcripciones de audio enviadas por "
            "cocineros o administradores y extraer con precisión quirúrgica cada producto, "
            "su cantidad numérica y la unidad de medida señalada. "
            "Si un producto no especifica unidad clara pero sí cantidad, infiere la unidad más "
            "lógica o usa 'unidad'. Extrae únicamente los insumos reales que constituyen una compra."
        )

    def validar_entrada(self, texto_pedido: str) -> None:
        """Valida que la entrada del usuario no esté vacía ni contenga tipos inválidos."""
        if not isinstance(texto_pedido, str):
            raise ValueError(f"La entrada debe ser de tipo str, se recibió: {type(texto_pedido).__name__}")
        if not texto_pedido or not texto_pedido.strip():
            raise ValueError("El texto del pedido no puede estar vacío ni contener solo espacios.")

    def calcular_costo_tokens(self, prompt_tokens: int, candidates_tokens: int) -> Dict[str, Any]:
        """Calcula el costo económico estimado de la inferencia."""
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
        Envía el texto informal a Gemini forzando salida JSON estructurada
        según el esquema PedidoExtraido y audita métricas de tokens y costos.
        """
        # Validación temprana (Criterio de Calidad de Código U1.3 / U1.4)
        self.validar_entrada(texto_pedido)

        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                response_mime_type="application/json",
                response_schema=PedidoExtraido,
                temperature=0.1,  # Baja temperatura para máxima fidelidad y consistencia
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Extrae los insumos del siguiente pedido:\n\n\"{texto_pedido.strip()}\"",
                config=config,
            )

            # Extraer uso de tokens para auditoría de costos
            usage = response.usage_metadata
            p_tokens = getattr(usage, "prompt_token_count", 0) or 0
            c_tokens = getattr(usage, "candidates_token_count", 0) or 0
            metricas_costo = self.calcular_costo_tokens(p_tokens, c_tokens)

            # Interpretar salida structured
            if response.parsed:
                datos_pedido = response.parsed.model_dump()
            else:
                texto_limpio = response.text.strip()
                texto_limpio = re.sub(r"^```(?:json)?\s*", "", texto_limpio)
                texto_limpio = re.sub(r"\s*```$", "", texto_limpio)
                datos_pedido = json.loads(texto_limpio)

            # Validación adicional con Pydantic si vino por texto crudo
            validado = PedidoExtraido(**datos_pedido).model_dump()

            return {
                "status": "success",
                "pedido": validado,
                "token_metrics": metricas_costo,
                "model_used": self.model_name
            }

        except APIError as api_err:
            return {
                "status": "error",
                "error_type": "Gemini_API_Error",
                "message": f"Error devuelto por la API del proveedor: {api_err}"
            }
        except json.JSONDecodeError as json_err:
            return {
                "status": "error",
                "error_type": "JSON_Decode_Error",
                "message": f"No se pudo decodificar la salida como JSON válido: {json_err}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "Unexpected_Error",
                "message": f"Falla inesperada durante el procesamiento: {e}"
            }


# ==========================================
# 3. Prueba de Ejecución Local
# ==========================================
if __name__ == "__main__":
    mensaje_ejemplo = (
        "Hola don Carlos, para mañana necesitamos con urgencia 2 cajas de tomate limachino "
        "bien firme, 3 sacos de harina panadera, unos 10 kilos de posta negra limpia "
        "y 2 bidones de lavaloza concentrado. Ah y si tiene palta hass, mándeme 1 caja. Gracias!"
    )

    print("=" * 60)
    print("Iniciando extracción con Gemini (OrderParser HORECA)...")
    print(f"Mensaje de entrada:\n{mensaje_ejemplo}")
    print("=" * 60)

    parser = OrderParser()
    resultado = parser.parsear_texto(mensaje_ejemplo)

    print("\nResultado estructurado y métricas:")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))