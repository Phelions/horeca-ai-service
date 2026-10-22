"""
================================================================================
SCRIPT: prompts/test_consistencia.py
PROPÓSITO: Medición empírica y automática de consistencia de los prompts (U1.5).
================================================================================

¿QUÉ HACE ESTE SCRIPT?
----------------------
Un error común en IA generativa es asumir que porque un prompt funcionó una vez,
siempre va a funcionar. Los modelos de lenguaje son probabilísticos.
Este script toma cada uno de los 3 prompts definidos en `prompts/banco.py` y los
ejecuta varias veces consecutivas (por defecto 3 repeticiones por prompt) contra
la API real de Gemini.

¿CÓMO EVALÚA EL ÉXITO?
----------------------
Para cada llamada:
1. Comprueba que la conexión no se caiga.
2. Comprueba que el texto devuelto se pueda convertir a JSON válido sin romperse.
3. Comprueba que el JSON contenga todas las claves requeridas (ej: 'items').
4. Si pasa los 3 filtros, suma 1 éxito.
5. Al final, guarda un archivo estructurado en `prompts/registro_resultados.json`
   demostrando la tasa de éxito (ej: 100%).

¿POR QUÉ TIENE UNA PAUSA (time.sleep)?
--------------------------------------
Para no saturar la cuota gratuita de Google AI Studio (error HTTP 429 de límite
de peticiones por minuto).
"""

# ==============================================================================
# IMPORTACIÓN DE LIBRERÍAS (¿Para qué sirve cada una?)
# ==============================================================================

# 'os' (Operating System): Nos permite consultar variables y rutas del sistema operativo.
import os

# 'sys' (System Parameters & Functions): Controla aspectos internos del intérprete de Python.
# Lo usamos aquí para indicarle a Python dónde está la carpeta raíz del proyecto (sys.path.insert)
# de modo que podamos hacer 'from src...' sin errores, sin importar desde qué carpeta ejecutes el comando.
import sys

# 'time': Funciones de fecha y medición de tiempo.
# Usamos 'time.sleep()' para hacer una pausa prudente entre llamadas a la IA y no saturar
# la cuota gratuita de Google AI Studio (evitando el molesto error HTTP 429 de demasiadas peticiones).
import time

# 'json' (JavaScript Object Notation): Nos permite manipular datos estructurados.
# Lo usamos para guardar el reporte final con los porcentajes de consistencia en 'prompts/registro_resultados.json'.
import json

# 'pathlib.Path': Permite manejar rutas de archivos de forma moderna y compatible con Windows, Mac y Linux.
# Lo usamos para calcular dinámicamente la ubicación exacta de la raíz del proyecto.
from pathlib import Path

# Añadir la carpeta raíz del proyecto a la lista de búsqueda de Python
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Importamos las 4 funciones modulares de nuestro propio cliente HTTP (creadas en src/consumo_api.py):
# - armar_peticion: Construye la URL y el cuerpo del mensaje para la API.
# - enviar_peticion: Hace la llamada HTTP real por internet y atrapa caídas de red o claves inválidas.
# - extraer_texto: Extrae el texto limpio de la respuesta de Gemini sin metadata técnica.
# - parsear_respuesta_json: Convierte el texto JSON en un diccionario de Python y elimina bloques markdown.
from src.consumo_api import armar_peticion, enviar_peticion, extraer_texto, parsear_respuesta_json

# Importamos la lista de prompts oficiales desde prompts/banco.py para ponerlos a prueba uno a uno
from prompts.banco import BANCO_DE_PROMPTS


def ejecutar_pruebas_consistencia(repeticiones_por_prompt: int = 3, pausa_segundos: float = 1.5):
    """
    Ejecuta el ciclo de pruebas de consistencia y genera el archivo JSON de métricas.
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
                # 1. Armamos la petición con la instrucción y el esquema estricto
                url, body = armar_peticion(
                    pregunta_usuario=prompt_info["ejemplo_usuario"],
                    system_instruction=prompt_info["system_instruction"],
                    response_schema=schema
                )
                
                # 2. Enviamos la llamada a Google Gemini
                respuesta_cruda = enviar_peticion(url, body)
                texto_generado = extraer_texto(respuesta_cruda)
                
                # 3. Limpiamos markdown y convertimos a JSON
                datos_parseados = parsear_respuesta_json(texto_generado)

                # 4. Validamos que estén todas las claves que el negocio necesita
                claves_faltantes = [c for c in claves_esperadas if c not in datos_parseados]
                if claves_faltantes:
                    raise ValueError(f"Faltan claves requeridas en el JSON: {claves_faltantes}")

                exitos += 1
                total_exitos += 1
                print("OK (JSON Válido y Estructura Conforme)")

            except Exception as err:
                print(f"FALLÓ: {err}")

            # Pausa preventiva de 1.5 segundos para respetar los límites de cuota (RPM)
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
