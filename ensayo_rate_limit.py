"""
ensayo_rate_limit.py - Ensayo controlado de límite de uso (Bloque 13).

Envía 10 peticiones seguidas contra la API para provocar y observar en vivo:
1. La captura del error de rate limit (HTTP 429).
2. El backoff exponencial con jitter espaciando los reintentos.
3. El registro y acumulación del costo de tokens en tiempo real.
"""

import cliente
from cliente import consultar_modelo


def main():
    print("Iniciando ensayo controlado de rate limit (10 consultas seguidas)...")
    print("------------------------------------------------------------------")

    for i in range(10):
        pregunta = f"Pregunta {i}: ¿Cuál es la fórmula del agua? Responde en una sola palabra."
        try:
            respuesta = consultar_modelo(pregunta)
            print(f"Respuesta {i}: {respuesta.strip()[:40]}...")
        except RuntimeError as error:
            print(f"Respuesta {i} falló definitivamente: {error}")

    print("------------------------------------------------------------------")
    print(f"Costo total del ensayo acumulado: ${cliente.costo_total_sesion:.5f} USD")


if __name__ == "__main__":
    main()
