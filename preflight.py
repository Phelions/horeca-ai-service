"""
preflight.py - Chequeo automático previo del flujo RAG sin llamadas a API (Bloque 28 / U2.5).

Verifica localmente que la carpeta del vector store persistido exista y esté declarada
en .gitignore antes de ejecutar consultas o evaluaciones reales.
"""

import os
from typing import List


def preflight_check(
    persist_directory: str = "vectorstore/",
    gitignore_path: str = ".gitignore",
) -> List[str]:
    """Confirma, sin llamar a ninguna API, que lo básico del flujo RAG está en orden."""
    problemas: List[str] = []
    if not os.path.isdir(persist_directory):
        problemas.append(f"{persist_directory} no existe: corriste la indexacion?")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, encoding="utf-8") as archivo:
            contenido = archivo.read()
        nombre_carpeta = os.path.basename(persist_directory.rstrip("/")) or persist_directory.rstrip("/")
        if nombre_carpeta not in contenido:
            problemas.append(f"{persist_directory} no esta en {gitignore_path}")
    else:
        problemas.append(f"no existe {gitignore_path} en este proyecto")
    return problemas


if __name__ == "__main__":
    problemas_detectados = preflight_check("vectorstore/")
    if problemas_detectados:
        print("Preflight con problemas:")
        for problema in problemas_detectados:
            print(" -", problema)
    else:
        print("Preflight OK: la carpeta del indice existe y esta en .gitignore")
