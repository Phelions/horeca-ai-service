# Registro del Ensayo de Rate Limit (Módulo 2 - Bloque 13)

Bitácora del ensayo empírico realizado con `ensayo_rate_limit.py`, provocando saturación controlada de peticiones por minuto contra la API de Gemini 3.8 Flash.

---

## 📋 Ficha de Observación del Ensayo

- **Fecha:** 2026-10-27
- **Peticiones enviadas:** 10 peticiones automáticas consecutivas.
- **Petición donde apareció el primer 429:** Petición número 6.
- **Esperas observadas:** 2.1s, 4.3s, 8.2s (backoff exponencial con jitter del 10%, comportamiento correcto).
- **Costo total del ensayo:** $0.00042 USD (~210 tokens en total).
- **Resultado:** Todas las peticiones terminaron con éxito tras reintentar automáticamente; ninguna excepción no controlada interrumpió el programa.

---

## 💡 Observaciones Clave

1. **Backoff Exponencial vs. Fijo:** Con esperas crecientes (2s, 4s, 8s...), la ventana de cuota del proveedor se libera antes de colisionar de nuevo.
2. **Jitter:** La pequeña variación aleatoria evita que múltiples llamadas concurrentes choquen al mismo segundo.
3. **Control de Costos:** Solo las respuestas que llegaron con código 200 sumaron tokens a `costo_total_sesion`.
