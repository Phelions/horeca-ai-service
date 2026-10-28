"""
ensayo_rate_limit.py - Ensayo controlado de límite de uso en HORECA AI Service (Bloque 13).

Simula el escenario típico de alta concurrencia al cierre del turno de cocina:
Se envían 10 pedidos gastronómicos seguidos sin pausas manuales para observar
en vivo el comportamiento del backoff exponencial con jitter ante posibles códigos HTTP 429,
asegurando que las compras no se pierdan y auditando el costo total en USD.
"""

import cliente
from cliente import consultar_modelo

# Lote de 10 pedidos gastronómicos reales de restaurante
PEDIDOS_HORECA_ENSAYO = [
    "Pedido 0 (Verdulería): 2 cajas de tomate limachino y 1 saco de cebolla morada.",
    "Pedido 1 (Carnicería): 10 kilos de posta negra cortada para churrasco y 5 kilos de lomo liso.",
    "Pedido 2 (Abarrotes): 3 sacos de harina de 25kg y 2 bidones de aceite vegetal de 5L.",
    "Pedido 3 (Lácteos): 10 barras de queso mozzarella de 3kg y 4 cajas de leche entera.",
    "Pedido 4 (Pescadería): 8 kilos de filete de reineta fresca y 3 kilos de salmón.",
    "Pedido 5 (Aseo Industrial): 3 bidones de lavaloza concentrado y 2 cajas de toallas de papel.",
    "Pedido 6 (Panadería): 40 unidades de pan marraqueta y 30 panes de hamburguesa brioche.",
    "Pedido 7 (Frutas): 1 caja de limones sutiles, 5 kilos de frutillas y 2 cajas de palta hass.",
    "Pedido 8 (Bebidas y Bar): 2 cajones de pisco transparente y 3 sifones de agua con gas.",
    "Pedido 9 (Especias y Granos): 1 kilo de pimienta negra entera, 2 kilos de orégano y 5kg de sal fina.",
]


def ejecutar_ensayo():
    print("=" * 72)
    print(" 🍳 HORECA AI SERVICE - ENSAYO CONTROLADO DE RATE LIMIT (10 PEDIDOS)")
    print("=" * 72)
    print("Enviando ráfaga consecutiva de pedidos para probar resiliencia y backoff...\n")

    for i, pedido in enumerate(PEDIDOS_HORECA_ENSAYO):
        pregunta = f"{pedido} Por favor confirma los insumos en una línea breve."
        try:
            respuesta = consultar_modelo(pregunta)
            print(f"✅ Pedido {i} procesado: {respuesta.strip()[:60]}...")
        except RuntimeError as error:
            print(f"❌ Pedido {i} falló definitivamente tras agotar reintentos: {error}")

    print("\n" + "=" * 72)
    print(f"💰 Costo total del ensayo acumulado: ${cliente.costo_total_sesion:.5f} USD")
    print("=" * 72)


if __name__ == "__main__":
    ejecutar_ensayo()
