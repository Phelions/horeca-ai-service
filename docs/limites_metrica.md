# Límites de la métrica de evaluación

## Qué mide

Mide la coincidencia léxica superficial de palabras (unigramas) entre el pedido gastronómico generado por el modelo y la orden de compra de referencia, considerando la frecuencia de repetición de cada término mediante bolsas de palabras (`Counter`), sintetizado en precisión, cobertura (*recall*) y su media armónica (*F1-Score*).


## Modos de falla conocidos

1. **Falsos negativos por sinónimos y jerga culinaria:**
   - La métrica castiga fuertemente términos equivalentes en cocina.
   - *Ejemplo real de revisión:* En el pedido de carnicería, la referencia esperaba *"posta negra limpia cortada en láminas para churrasco"* y el modelo respondió *"carne de vacuno desgrasada en bistec delgado para sándwich"*. La orden es 100% válida para abastecimiento, pero el F1 cayó a **0.214** (21.4%).

2. **Falsos positivos por errores numéricos en cantidades o calibres:**
   - Una variación de un solo dígito altera marginalmente el vocabulario, pero altera catastróficamente la compra mayorista.
   - *Ejemplo real de revisión:* En el pedido de verdulería, la referencia esperaba *"2 cajas de palta hass madura"* y la respuesta confirmó *"20 cajas de palta hass madura"*. El F1 resultante fue de **0.933** (93.3%), ocultando un sobrecosto de 10x y riesgo de merma total por sobremaduración.

3. **Inversión de significado por negaciones ausentes (Riesgo Bromatológico y Alérgenos):**
   - La omisión o adición de una partícula de negación (*"no"*, *"sin"*) casi no altera el puntaje global, pero invierte el sentido crítico del alimento.
   - *Ejemplo real de revisión:* Consulta sobre alérgenos: la referencia indicaba *"no contiene gluten de trigo"* y la respuesta generó *"contiene gluten de trigo"*. El F1 obtenido fue de **0.952** (95.2%), calificando como casi perfecta una afirmación que induce riesgo médico para clientes celíacos.

4. **Omisión de especificaciones de cadena de frío y perecibilidad:**
   - Omitir atributos de temperatura o estado no penaliza severamente el F1 al ser pocas palabras entre muchas que coinciden.
   - *Ejemplo real de revisión:* En camarón ecuatoriano, se omitió *"congelado con hielo seco"*, logrando un F1 de **0.818** a pesar de violar el protocolo sanitario de transporte refrigerado.

5. **Discrepancia en formatos de envasado y bodegaje:**
   - Reemplazar sacos de 50 kg por sacos de 25 kg o bidones de 20L por botellas de 1L arroja F1 moderado (~0.64), pero afecta el espacio físico de bodega y la ergonomía de descarga en cocina.


## Muestra revisada

Se analizaron **10 casos estratificados** (los 5 con menor F1 y los 5 con mayor F1) sobre un universo de **20 órdenes gastronómicas reales** generadas y evaluadas el 2026-10-30 en `resultados_metricas.json`, aplicando una rúbrica cualitativa de tres dimensiones:
- **Corrección factual:** ¿Cantidades, productos y especificaciones son exactos?
- **Completitud:** ¿Se abarcó la totalidad de insumos sin omisiones críticas?
- **Contradicción:** ¿Se contradijo alguna indicación de stock o alérgenos?


## Recomendación de uso y Métrica Ideal para HORECA

1. **Uso de F1 Léxico:**
   - Utilizar F1 exclusivamente como **filtro automatizado de screening masivo** para comparar versiones de prompts o modelos a gran escala y detectar respuestas desalineadas o alucinaciones severas.

2. **Por qué F1 solo no basta en HORECA:**
   - En hotelería y restaurantes, la información transaccional es crítica: un error en un número de comensales, una cantidad de kilos o un alérgeno invalida la orden comercial. F1 trata la palabra *"tomate"* con el mismo peso matemático que el número *"20"*.

3. **Métrica Ideal Propuesta (Enfoque Híbrido para Producción):**
   - **Evaluación Estructurada basada en Entidades (NER F1):** Medir precisión y cobertura sobre entidades de compra (Producto, Cantidad, Unidad, Detalle).
   - **Exact Match (EM) en Cantidades y Unidades:** La cantidad numérica y unidad de medida deben coincidir al 100% de forma determinística con esquemas Pydantic.
   - **Inspección manual obligatoria:** Auditar siempre a mano cualquier respuesta que involucre cortes específicos de carnes, mariscos frescos, o advertencias de alérgenos.

