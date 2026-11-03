"""
pipeline_texto.py - Pipeline de procesamiento, limpieza y evaluación léxica de salidas (U2.2).

Contiene las funciones atómicas para:
1. Normalizar texto (minúsculas, sin puntuación ni espacios redundantes).
2. Tokenizar cadenas limpias en listas de palabras.
3. Centralizar el procesamiento de texto (limpiar + tokenizar).
4. Vectorizar mediante conteos de frecuencias (bolsa de palabras con Counter).
5. Calcular métricas formales de precisión, cobertura (recall) y F1.
6. Resumir conjuntos de evaluaciones para análisis de calidad en HORECA AI Service.
"""

import re
from collections import Counter
from typing import List, Dict, Any


def limpiar_texto(texto: str) -> str:
    """
    Normaliza el texto eliminando ruido de formato:
    - Valida que la entrada sea de tipo str (lanza TypeError si no lo es).
    - Convierte todo a minúsculas.
    - Remueve signos de puntuación y caracteres especiales preservando tildes y la 'ñ' (\\w).
    - Colapsa espacios múltiples en uno solo y aplica strip().
    """
    if not isinstance(texto, str):
        tipo = type(texto).__name__
        raise TypeError(f"limpiar_texto espera str, no {tipo}")

    texto = texto.lower()
    # [^\w\s] elimina todo lo que no sea alfanumérico ni espacio en blanco
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto_limpio: str) -> List[str]:
    """
    Convierte una cadena de texto ya normalizada en una lista de tokens (palabras).
    Separa por espacios en blanco. Si la cadena está vacía, retorna lista vacía.
    """
    return texto_limpio.split()


def procesar_texto(texto: str) -> List[str]:
    """
    Pipeline unificado: centraliza la limpieza y la tokenización en un solo paso
    para garantizar que las comparaciones nunca se realicen sobre texto crudo desalineado.
    """
    texto_limpio = limpiar_texto(texto)
    return tokenizar(texto_limpio)


def vectorizar(tokens: List[str]) -> Counter:
    """
    Representación bolsa de palabras (Bag of Words):
    Genera un vector de frecuencias contando cuántas veces aparece cada token.
    """
    return Counter(tokens)


def contar_coincidencias(
    tokens_referencia: List[str],
    tokens_respuesta: List[str],
) -> int:
    """
    Cuenta las coincidencias exactas entre tokens de referencia y respuesta sin perder repeticiones.
    Utiliza la intersección de Counter (&), que toma el mínimo de apariciones en ambos lados.
    """
    conteo_referencia = Counter(tokens_referencia)
    conteo_respuesta = Counter(tokens_respuesta)
    interseccion = conteo_referencia & conteo_respuesta
    return sum(interseccion.values())


def calcular_precision_cobertura_f1(
    tokens_referencia: List[str],
    tokens_respuesta: List[str],
) -> Dict[str, float]:
    """
    Calcula precisión, cobertura (recall) y F1-Score entre dos listas de tokens:
    - Precisión: coincidencias / total de tokens en la respuesta (¿cuánto de lo dicho es correcto?).
    - Cobertura: coincidencias / total de tokens en la referencia (¿cuánto de lo esperado se cubrió?).
    - F1: media armónica balanceada entre precisión y cobertura.
    Maneja con seguridad casos borde donde alguna de las listas esté vacía (retorna 0.0).
    """
    coincidencias = contar_coincidencias(tokens_referencia, tokens_respuesta)

    if not tokens_respuesta or not tokens_referencia:
        return {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}

    precision = coincidencias / len(tokens_respuesta)
    cobertura = coincidencias / len(tokens_referencia)

    if precision + cobertura == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * cobertura / (precision + cobertura)

    return {
        "precision": round(precision, 4),
        "cobertura": round(cobertura, 4),
        "f1": round(f1, 4),
    }


def calcular_metricas(texto_referencia: str, texto_respuesta: str) -> Dict[str, float]:
    """
    Orquestador de comparación directa: procesa ambos textos con el pipeline
    y calcula precisión, cobertura y F1 en una sola llamada.
    """
    tokens_ref = procesar_texto(texto_referencia)
    tokens_resp = procesar_texto(texto_respuesta)
    return calcular_precision_cobertura_f1(tokens_ref, tokens_resp)


def resumir_metricas(resultados: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula los promedios globales de precisión, cobertura y F1 sobre una lista de evaluaciones.
    """
    if not resultados:
        return {
            "precision_promedio": 0.0,
            "cobertura_promedio": 0.0,
            "f1_promedio": 0.0,
        }

    n = len(resultados)
    return {
        "precision_promedio": round(sum(r["precision"] for r in resultados) / n, 4),
        "cobertura_promedio": round(sum(r["cobertura"] for r in resultados) / n, 4),
        "f1_promedio": round(sum(r["f1"] for r in resultados) / n, 4),
    }


if __name__ == "__main__":
    print("=" * 72)
    print(" 🍳 HORECA AI SERVICE - DEMOSTRACIÓN DEL PIPELINE DE TEXTO (BLOQUE 15)")
    print("=" * 72)

    # 3 Casos de prueba reales del dominio gastronómico para ilustrar el impacto del pipeline
    casos_demostracion = [
        (
            "Pedido Verdulería",
            "Necesitamos 2 cajas de tomate limachino bien firme y 1 saco de cebolla.",
            "necesitamos 2 cajas de tomate limachino bien firme y 1 saco de cebolla"
        ),
        (
            "Pedido Carnicería",
            "Mándame 10 kilos de posta negra limpia, cortada para churrasco.",
            "mandame 10 kilos de posta negra limpia cortada para churrasco"
        ),
        (
            "Pedido Abarrotes",
            "Confirmar 3 sacos de harina panadera de 25kg y 2 bidones de aceite vegetal.",
            "Confirmar 3 sacos de harina panadera de 25kg y 2 bidones de aceite vegetal."
        ),
    ]

    for nombre, referencia, respuesta in casos_demostracion:
        # 1. Comparación ingenua sobre texto crudo (split directo)
        crudo_ref = referencia.split()
        crudo_resp = respuesta.split()
        comunes_crudo = set(crudo_ref) & set(crudo_resp)
        pct_crudo = len(comunes_crudo) / len(set(crudo_ref)) if crudo_ref else 0.0

        # 2. Comparación pasando por procesar_texto
        tokens_ref = procesar_texto(referencia)
        tokens_resp = procesar_texto(respuesta)
        comunes_proc = set(tokens_ref) & set(tokens_resp)
        pct_proc = len(comunes_proc) / len(set(tokens_ref)) if tokens_ref else 0.0

        metricas_f1 = calcular_precision_cobertura_f1(tokens_ref, tokens_resp)

        print(f"\n--- {nombre} ---")
        print(f"Texto Referencia : \"{referencia}\"")
        print(f"Texto Respuesta  : \"{respuesta}\"")
        print(f"❌ Coincidencia Texto Crudo     : {pct_crudo:.1%}")
        print(f"✅ Coincidencia Texto Procesado : {pct_proc:.1%}")
        print(f"📊 Métricas F1 (Bolsa de palabras): Precisión={metricas_f1['precision']}, Cobertura={metricas_f1['cobertura']}, F1={metricas_f1['f1']}")

