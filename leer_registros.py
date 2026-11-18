"""
leer_registros.py - Lector resumido de trazas de latencia y costo RAG (Bloque 28 / U2.5).

Lee registros_latencia_costo.jsonl línea por línea (JSON Lines en UTF-8) y muestra
k, input_tokens, tiempo_total_ms y el inicio de cada pregunta para comparar corridas.
"""

import json
from typing import List


def resumir_registros(ruta_archivo: str = "registros_latencia_costo.jsonl") -> List[str]:
    """Lee el archivo JSONL de latencia/costo y devuelve líneas formateadas de resumen."""
    lineas_resumen: List[str] = []
    with open(ruta_archivo, encoding="utf-8") as archivo:
        for linea in archivo:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            tokens_entrada = (
                registro["tokens"]["input_tokens"] if registro.get("tokens") else None
            )
            lineas_resumen.append(
                f"k={registro['k']}  input_tokens={tokens_entrada}  "
                f"total={registro['tiempo_total_ms']} ms  {registro['pregunta'][:35]}"
            )
    return lineas_resumen


if __name__ == "__main__":
    for linea_salida in resumir_registros():
        print(linea_salida)
