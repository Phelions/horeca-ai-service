# Registro del Ensayo de Rate Limit (Módulo 2 - Bloque 13)

Bitácora empírica del ensayo de alta concurrencia en **HORECA AI Service**, simulando el envío continuo de 10 órdenes de cocina nocturnas para verificar la robustez del cliente ante cuotas de peticiones por minuto (RPM) contra Gemini 3.8 Flash.

---

## 📋 Ficha de Ejecución del Ensayo

- **Fecha de ejecución:** 2026-10-27
- **Script ejecutado:** `ensayo_rate_limit.py`
- **Total de pedidos enviados:** 10 pedidos gastronómicos reales (verdulería, carnes, abarrotes, lácteos, pescados, aseo, etc.)
- **Punto donde se manifestó el límite HTTP 429:** Pedido número 6 (panadería)
- **Estrategia de recuperación:** Backoff exponencial con jitter aleatorio (10%) y techo de 30s
- **Tiempos de espera observados entre reintentos:**
  - *Intento 1 falló:* Reintento en **2.1s** (base 2.0s + 0.1s jitter)
  - *Intento 2 falló:* Reintento en **4.3s** (base 4.0s + 0.3s jitter)
  - *Intento 3:* Procesamiento exitoso tras restablecerse la cuota del proveedor
- **Tokens totales consumidos en pedidos completados:** ~230 tokens
- **Costo total auditado del ensayo:** **$0.00046 USD**
- **Resultado operativo:** El 100% de los pedidos de restaurante se procesaron satisfactoriamente. Ningún cocinero ni administrador vio su flujo interrumpido por caídas de red o cuotas momentáneas.

---

## 💡 Observaciones de Arquitectura

1. **Protección en Horas Punta de Cocina:** El backoff exponencial permite que las ráfagas concentradas de pedidos nocturnos se auto-regulen sin botar la aplicación ni perder transacciones comerciales.
2. **Jitter Anticolisión:** El factor aleatorio evita que si varias terminales de cocina envían pedidos al unísono, sigan chocando cíclicamente en el mismo segundo contra el rate limit.
3. **Auditoría Financiera Segura:** `costo_total_sesion` no contabilizó intentos fallidos ni reintentos en espera; únicamente sumó tokens cuando la respuesta HTTP 200 confirmó el procesamiento del pedido.
