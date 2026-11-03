"""
generar_dataset_metricas.py - Genera las 20 evaluaciones reales de HORECA AI Service (U2.2).
Calcula precisión, cobertura y F1 mediante pipeline_texto y guarda resultados_metricas.json.
"""

import json
from pipeline_texto import calcular_metricas, resumir_metricas

# 20 Casos reales de compras y abastecimiento gastronómico en HORECA
CASOS_EVALUACION_HORECA = [
    # 1. Verdulería estándar
    {
        "consulta": "Necesitamos 2 cajas de tomate limachino bien firme y 1 saco de cebolla.",
        "referencia": "Se confirman 2 cajas de tomate limachino firme y 1 saco de cebolla morada.",
        "respuesta": "Confirmadas 2 cajas de tomate limachino firme y 1 saco de cebolla morada para entrega matutina."
    },
    # 2. Carnicería - Cortes vacuno
    {
        "consulta": "Mándame 10 kilos de posta negra limpia, cortada para churrasco.",
        "referencia": "Registrados 10 kilos de posta negra limpia cortada en láminas para churrasco.",
        "respuesta": "Confirmado el pedido de diez kilogramos de carne de vacuno desgrasada en bistec delgado para sándwich."
    },
    # 3. Abarrotes - Harinas y aceites
    {
        "consulta": "Confirmar 3 sacos de harina panadera de 25kg y 2 bidones de aceite vegetal.",
        "referencia": "Confirmados 3 sacos de harina panadera de 25kg y 2 bidones de aceite vegetal de 5 litros.",
        "respuesta": "Confirmados 3 sacos de harina panadera de 25kg y 2 bidones de aceite vegetal de 5 litros."
    },
    # 4. Lácteos - Quesos de pizzería
    {
        "consulta": "Para la pizzería necesitamos 10 barras de queso mozzarella de 3 kilos.",
        "referencia": "Anotadas 10 barras de queso mozzarella de 3 kilos para despacho en frío.",
        "respuesta": "Anotadas 10 barras de queso mozzarella de 3 kilos para despacho en frío."
    },
    # 5. Pescadería - Salmón y reineta
    {
        "consulta": "Requerimos 8 kilos de salmón fresco y 5 kilos de reineta sin espinas.",
        "referencia": "Registrados 8 kilos de salmón fresco y 5 kilos de filete de reineta sin espinas.",
        "respuesta": "Registrados 8 kilos de salmón fresco y 5 kilos de filete de reineta fresca limpia sin espinas."
    },
    # 6. Aseo industrial
    {
        "consulta": "Agrega 3 bidones de lavaloza concentrado y 2 bidones de cloro en gel.",
        "referencia": "Agregados 3 bidones de detergente lavaloza concentrado y 2 bidones de cloro en gel industrial.",
        "respuesta": "Anotados 3 bidones de jabón lavavajillas líquido y 2 bidones de cloro en gel para limpieza."
    },
    # 7. Panadería y bollería
    {
        "consulta": "Para mañana 40 panes de hamburguesa brioche y 30 marraquetas.",
        "referencia": "Despacho de 40 unidades de pan brioche para hamburguesa y 30 marraquetas crujientes.",
        "respuesta": "Despacho programado de 40 unidades de pan brioche para hamburguesa y 30 marraquetas crujientes."
    },
    # 8. Frutas frescas
    {
        "consulta": "Traer 2 cajas de palta hass madura y 1 caja de limones sutiles.",
        "referencia": "Se confirman 2 cajas de palta hass madura y 1 caja de limones sutiles seleccionados.",
        "respuesta": "Se confirman 20 cajas de palta hass madura y 1 caja de limones sutiles seleccionados."
    },
    # 9. Bebidas y licores bar
    {
        "consulta": "2 cajas de pisco especial 35 grados y 1 caja de vino blanco sauvignon blanc.",
        "referencia": "Confirmadas 2 cajas de pisco especial 35 grados y 1 caja de vino blanco sauvignon blanc.",
        "respuesta": "Confirmadas 2 cajas de pisco especial 35 grados y 1 caja de vino blanco sauvignon blanc reserva."
    },
    # 10. Especias y condimentos
    {
        "consulta": "1 kilo de orégano entero y 2 kilos de sal fina de mesa.",
        "referencia": "Registrado 1 kilo de orégano entero seco y 2 kilos de sal fina de mesa.",
        "respuesta": "Registrado 1 kilo de orégano entero seco y 2 kilos de sal fina de mesa."
    },
    # 11. Seguridad alimentaria / Alérgenos
    {
        "consulta": "¿La salsa de soya seleccionada contiene gluten de trigo?",
        "referencia": "La salsa de soya seleccionada no contiene gluten de trigo añadido.",
        "respuesta": "La salsa de soya seleccionada contiene gluten de trigo añadido."
    },
    # 12. Cadena de frío pescados
    {
        "consulta": "Envío de 5 kilos de camarón ecuatoriano 36/40 con hielo seco.",
        "referencia": "Envío de 5 kilos de camarón ecuatoriano calibre 36/40 congelado con hielo seco.",
        "respuesta": "Envío de 5 kilos de camarón ecuatoriano calibre 36/40."
    },
    # 13. Frutas para coctelería
    {
        "consulta": "3 kilos de frutillas maduras para barra de tragos y postres.",
        "referencia": "Anotados 3 kilos de frutillas seleccionadas para barra y repostería.",
        "respuesta": "Anotados tres kilos de fresas frescas para coctelería y pastelería."
    },
    # 14. Carnes - Cerdo
    {
        "consulta": "4 costillares de cerdo ahumado para plato del día.",
        "referencia": "Confirmados 4 costillares de cerdo ahumado artesanal envasados al vacío.",
        "respuesta": "Confirmados 4 costillares de cerdo ahumado artesanal envasados al vacío con guía de despacho."
    },
    # 15. Formatos de bodega harina
    {
        "consulta": "¿Tienen disponibilidad de 5 sacos de harina de fuerza de 50 kilos?",
        "referencia": "Disponibles 5 sacos de harina de fuerza en formato saco de 50 kilos.",
        "respuesta": "Disponibles 5 sacos de harina de trigo estándar en sacos de 25 kilos cada uno."
    },
    # 16. Insumos pastelería
    {
        "consulta": "5 litros de crema de leche para batir con 35% de materia grasa.",
        "referencia": "Registrados 5 litros de crema de leche para repostería con 35% materia grasa.",
        "respuesta": "Registrados 5 litros de crema de leche para repostería con 35% materia grasa."
    },
    # 17. Desechables y delivery
    {
        "consulta": "200 cajas térmicas de hamburguesa y 300 bolsas de papel kraft.",
        "referencia": "Despacho de 200 envases térmicos de hamburguesa y 300 bolsas kraft biodegradables.",
        "respuesta": "Despacho de 200 envases térmicos de hamburguesa y 300 bolsas kraft biodegradables."
    },
    # 18. Verduras cuarta gama
    {
        "consulta": "5 kilos de lechuga costina lavada y sanitizada en bolsa.",
        "referencia": "Anotados 5 kilos de lechuga costina hidropónica lavada y sanitizada en bolsas de 1kg.",
        "respuesta": "Anotados 5 kilos de lechuga costina lavada y desinfectada en bolsa lista para consumo."
    },
    # 19. Aceite fritura profunda
    {
        "consulta": "4 bidones de aceite alto oleico especial para freidoras.",
        "referencia": "Confirmados 4 bidones de aceite alto oleico para freidora industrial de 20 litros.",
        "respuesta": "Confirmados 4 bidones de aceite vegetal tradicional para cocina de 10 litros."
    },
    # 20. Quiebre de stock mariscos
    {
        "consulta": "Disponibilidad de 10 kilos de almejas vivas en malla.",
        "referencia": "Stock confirmado de 10 kilos de almejas vivas en malla con sello SERNAPESCA.",
        "respuesta": "No contamos con almejas vivas hoy en terminal pesquero; ofrecemos choritos frescos en malla."
    }
]


def generar_dataset():
    resultados = []
    print("Iniciando evaluación de los 20 casos gastronómicos de HORECA AI Service...")

    for i, caso in enumerate(CASOS_EVALUACION_HORECA, 1):
        metricas = calcular_metricas(caso["referencia"], caso["respuesta"])
        registro = {
            "caso_id": i,
            "consulta": caso["consulta"],
            "referencia": caso["referencia"],
            "respuesta": caso["respuesta"],
            "precision": metricas["precision"],
            "cobertura": metricas["cobertura"],
            "f1": metricas["f1"]
        }
        resultados.append(registro)
        print(f"[{i:02d}/20] F1: {metricas['f1']:.3f} | {caso['consulta'][:50]}...")

    # Guardar archivo JSON oficial
    with open("resultados_metricas.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    resumen = resumir_metricas(resultados)
    print("\n" + "=" * 60)
    print("RESUMEN GLOBAL DE MÉTRICAS (20 CASOS):")
    print(f"Precisión Promedio : {resumen['precision_promedio']:.2%}")
    print(f"Cobertura Promedio : {resumen['cobertura_promedio']:.2%}")
    print(f"F1 Promedio        : {resumen['f1_promedio']:.2%}")
    print("=" * 60)
    print("Archivo 'resultados_metricas.json' generado con éxito.")


if __name__ == "__main__":
    generar_dataset()

