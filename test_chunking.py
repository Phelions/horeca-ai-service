"""
test_chunking.py - Comparación de estrategias de chunking en vivo (Unidad 2.5 / Bloque 24).

Compara CharacterTextSplitter (tamaño fijo sin mirar estructura) contra
RecursiveCharacterTextSplitter (respetando párrafos, oraciones y palabras) sobre un
texto del dominio HORECA, e incluye pruebas unitarias ejecutables con pytest.
"""

from typing import Dict, List

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


TEXTO_PRUEBA_HORECA: str = (
    "El cliente B2B HORECA reintenta hasta 3 veces ante un TimeoutError, con espera "
    "creciente por intento.\n\n"
    "Los errores de autenticacion nunca se reintentan: fallan al instante con un "
    "ValueError, sin espera.\n\n"
    "El costo de cada llamada a Gemini se registra por sesion mientras la plataforma "
    "sigue operando."
)


def comparar_estrategias_chunking(
    texto: str = TEXTO_PRUEBA_HORECA,
    chunk_size: int = 110,
    chunk_overlap_recursivo: int = 20,
) -> Dict[str, List[str]]:
    """Parte un texto con splitter fijo y splitter recursivo para comparar los chunks."""
    splitter_fijo = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    splitter_recursivo = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap_recursivo,
    )

    chunks_fijo = splitter_fijo.split_text(texto)
    chunks_recursivo = splitter_recursivo.split_text(texto)
    return {
        "fijo": chunks_fijo,
        "recursivo": chunks_recursivo,
    }


def test_splitter_fijo_corta_palabras():
    """El splitter fijo corta exactamente cada chunk_size caracteres aunque parta palabras."""
    resultado = comparar_estrategias_chunking()
    chunks_fijo = resultado["fijo"]

    assert len(chunks_fijo) == 3
    assert len(chunks_fijo[0]) == 110
    assert "Los err" in chunks_fijo[0]
    assert chunks_fijo[1].startswith("ores de autenticacion")


def test_splitter_recursivo_preserva_parrafos():
    """El splitter recursivo respeta los saltos de párrafo y conserva cada idea completa."""
    resultado = comparar_estrategias_chunking()
    chunks_recursivo = resultado["recursivo"]

    assert len(chunks_recursivo) == 3
    assert chunks_recursivo[0].startswith("El cliente B2B HORECA")
    assert chunks_recursivo[1].startswith("Los errores de autenticacion")
    assert chunks_recursivo[2].startswith("El costo de cada llamada")


if __name__ == "__main__":
    comparacion = comparar_estrategias_chunking()
    print("=== Estrategia 1: CharacterTextSplitter (tamaño fijo) ===")
    for indice, fragmento in enumerate(comparacion["fijo"]):
        print(f"--- chunk {indice} ({len(fragmento)} caracteres) ---")
        print(fragmento)

    print("\n=== Estrategia 2: RecursiveCharacterTextSplitter (recursivo) ===")
    for indice, fragmento in enumerate(comparacion["recursivo"]):
        print(f"--- chunk {indice} ({len(fragmento)} caracteres) ---")
        print(fragmento)
