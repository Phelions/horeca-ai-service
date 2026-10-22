"""
================================================================================
PUNTO DE ENTRADA PRINCIPAL: src/main.py
PROPÓSITO: Ejecución y demostración de la miniaplicación HORECA AI Service (Gate 1).
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
Es el punto de entrada oficial ("main") del servicio. Recibe un pedido gastronómico
informal (texto libre de cocina, WhatsApp o transcripción de audio), orquesta
el flujo de extracción estructurada mediante IA (Gemini 3.8 Flash), valida la salida
con esquemas Pydantic y muestra en consola el JSON procesado junto con la auditoría
de tokens y costos en dólares (USD).

¿CÓMO SE EJECUTA?
------------------
Desde la raíz del proyecto:
    python src/main.py

O como módulo de Python:
    python -m src.main
"""

import sys
import json
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
# para permitir la ejecución tanto desde la raíz como desde dentro de 'src/'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.order_parser import OrderParser
except ImportError:
    from order_parser import OrderParser


def main():
    """
    Función principal de ejecución del Gate 1:
    Demuestra la extracción y normalización de un pedido informal de restaurante.
    """
    mensaje_ejemplo = (
        "Hola don Carlos, para mañana necesitamos con urgencia 2 cajas de tomate limachino "
        "bien firme, 3 sacos de harina panadera, unos 10 kilos de posta negra limpia "
        "y 2 bidones de lavaloza concentrado. Ah y si tiene palta hass, mándeme 1 caja. Gracias!"
    )

    print("=" * 75)
    print(" 🍳 HORECA AI SERVICE — PROCESAMIENTO INTELIGENTE DE PEDIDOS (GATE 1)")
    print("=" * 75)
    print("\n[ENTRADA INFORMAL RECIBIDA DE COCINA]:")
    print(f"\"{mensaje_ejemplo}\"\n")
    print("-" * 75)
    print("Procesando con Gemini 3.8 Flash y validando con Pydantic...")

    parser = OrderParser()
    resultado = parser.parsear_texto(mensaje_ejemplo)

    print("-" * 75)
    print("RESULTADO ESTRUCTURADO (JSON):")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
