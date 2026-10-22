"""
================================================================================
PAQUETE: src/__init__.py
PROPÓSITO: Marcador de paquete de módulos para el código fuente del proyecto.
================================================================================

¿QUÉ HACE ESTE ARCHIVO?
-----------------------
En Python, la presencia de un archivo llamado '__init__.py' dentro de una carpeta
le indica al intérprete que este directorio debe ser tratado como un "paquete"
de código estructurado y reutilizable.

¿POR QUÉ ES NECESARIO?
----------------------
Permite que otros scripts (como los tests o las herramientas de línea de comandos)
puedan importar funciones y clases usando la sintaxis:
    `from src.order_parser import OrderParser`
    `from src.consumo_api import armar_peticion`
sin errores de módulos no encontrados (ModuleNotFoundError).
"""
