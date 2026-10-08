import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Cargar variables de entorno desde .env
load_dotenv()

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
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró GEMINI_API_KEY en las variables de entorno. "
                "Verifica tu archivo .env"
            )
        
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.client = genai.Client(api_key=api_key)
        
        self.system_instruction = (
            "Eres un asistente de adquisiciones para el sector gastronómico (HORECA). "
            "Tu misión es analizar mensajes informales o transcripciones de audio enviadas por "
            "cocineros o administradores y extraer con precisión quirúrgica cada producto, "
            "su cantidad numérica y la unidad de medida señalada. "
            "Si un producto no especifica unidad clara pero sí cantidad, infiere la unidad más "
            "lógica o usa 'unidad'. Extrae únicamente los insumos reales que constituyen una compra."
        )

    def parsear_texto(self, texto_pedido: str) -> dict:
        """
        Envía el texto informal a Gemini forzando salida JSON estructurada
        según el esquema PedidoExtraido y captura métricas de tokens.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                response_mime_type="application/json",
                response_schema=PedidoExtraido,
                temperature=0.1,  # Baja temperatura para máxima consistencia y fidelidad
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Extrae los insumos del siguiente pedido:\n\n\"{texto_pedido}\"",
                config=config,
            )

            # Extraer uso de tokens para auditoría de costos
            usage = response.usage_metadata
            token_metrics = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "candidates_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }

            # Si el SDK ya parseó el objeto con el esquema Pydantic
            if response.parsed:
                datos_pedido = response.parsed.model_dump()
            else:
                datos_pedido = json.loads(response.text)

            return {
                "status": "success",
                "pedido": datos_pedido,
                "token_metrics": token_metrics,
                "model_used": self.model_name
            }

        except APIError as api_err:
            return {
                "status": "error",
                "error_type": "Gemini_API_Error",
                "message": str(api_err)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "Unexpected_Error",
                "message": str(e)
            }


# ==========================================
# 3. Prueba de Ejecución Local
# ==========================================
if __name__ == "__main__":
    # Mensaje informal típico de un restaurante
    mensaje_ejemplo = (
        "Hola don Carlos, para mañana necesitamos con urgencia 2 cajas de tomate limachino "
        "bien firme, 3 sacos de harina panadera, unos 10 kilos de posta negra limpia "
        "y 2 bidones de lavaloza concentrado. Ah y si tiene palta hass, mándeme 1 caja. Gracias!"
    )

    print("=" * 60)
    print("Iniciando extracción con Gemini...")
    print(f"Mensaje de entrada:\n{mensaje_ejemplo}")
    print("=" * 60)

    parser = OrderParser()
    resultado = parser.parsear_texto(mensaje_ejemplo)

    print("\nResultado estructurado:")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))