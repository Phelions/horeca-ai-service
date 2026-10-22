import os
import sys
import time
import json
from pathlib import Path

# Agregar directorio raíz al PYTHONPATH para importar src
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.consumo_api import armar_peticion, enviar_peticion, extraer_texto, parsear_respuesta_json
from prompts.banco import BANCO_DE_PROMPTS


def ejecutar_pruebas_consistencia(repeticiones_por_prompt: int = 3, pausa_segundos: float = 1.5):
    """
    Ejecuta cada prompt del banco múltiples veces contra la API de Gemini
    y valida que la salida sea un JSON conforme con las claves esperadas.
    """
    print("=" * 70)
    print(" INICIANDO VALIDACIÓN DE CONSISTENCIA DE BANCO DE PROMPTS (U1.5)")
    print(f" Total de prompts en banco: {len(BANCO_DE_PROMPTS)}")
    print(f" Repeticiones por prompt:   {repeticiones_por_prompt}")
    print("=" * 70)

    registro_metricas = {}
    total_pruebas = 0
    total_exitos = 0

    for prompt_info in BANCO_DE_PROMPTS:
        nombre = prompt_info["nombre"]
        claves_esperadas = prompt_info["claves_esperadas"]
        schema = prompt_info.get("response_schema")
        
        print(f"\n[Evaluando Prompt]: '{nombre}'")
        exitos = 0

        for intento in range(repeticiones_por_prompt):
            total_pruebas += 1
            print(f"  -> Ejecutando intento {intento + 1}/{repeticiones_por_prompt}...", end=" ")
            
            try:
                url, body = armar_peticion(
                    pregunta_usuario=prompt_info["ejemplo_usuario"],
                    system_instruction=prompt_info["system_instruction"],
                    response_schema=schema
                )
                
                respuesta_cruda = enviar_peticion(url, body)
                texto_generado = extraer_texto(respuesta_cruda)
                datos_parseados = parsear_respuesta_json(texto_generado)

                # Validar presencia de todas las claves requeridas
                claves_faltantes = [c for c in claves_esperadas if c not in datos_parseados]
                if claves_faltantes:
                    raise ValueError(f"Faltan claves requeridas en el JSON: {claves_faltantes}")

                exitos += 1
                total_exitos += 1
                print("OK (JSON Válido y Estructura Conforme)")

            except Exception as err:
                print(f"FALLÓ: {err}")

            # Pausa preventiva para evitar error HTTP 429 (límite de cuota)
            time.sleep(pausa_segundos)

        porcentaje = (exitos / repeticiones_por_prompt) * 100
        print(f"  Resultado '{nombre}': {exitos}/{repeticiones_por_prompt} válidas ({porcentaje:.1f}% consistencia)")

        registro_metricas[nombre] = {
            "exitos": exitos,
            "intentos": repeticiones_por_prompt,
            "tasa_exito_porcentaje": round(porcentaje, 1),
            "claves_esperadas": claves_esperadas,
            "consistencia_robusta": exitos == repeticiones_por_prompt
        }

    # Guardar reporte estructurado en prompts/registro_resultados.json
    ruta_salida = ROOT_DIR / "prompts" / "registro_resultados.json"
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        json.dump(registro_metricas, archivo, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f" REPORTE FINAL: {total_exitos}/{total_pruebas} pruebas exitosas ({(total_exitos/total_pruebas)*100:.1f}%)")
    print(f" Registro guardado en: {ruta_salida}")
    print("=" * 70)

    return registro_metricas


if __name__ == "__main__":
    ejecutar_pruebas_consistencia()
