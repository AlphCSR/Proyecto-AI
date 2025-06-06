"""
Paquete principal de la aplicación de predicción de calidad de vinos.

Este paquete contiene los módulos principales de la aplicación:
- Interfaz gráfica (gui)
- Modelos de Machine Learning (models)
- Utilidades varias (utils)
- Punto de entrada principal (main.py)
"""

__version__ = '1.0.0'
__author__ = 'AlphCSR Team'
__email__ = 'contacto@alphcsr.com'
__license__ = 'MIT'

# Importaciones principales
from . import gui
from . import models
from . import utils

# Hacer que estas importaciones estén disponibles en el espacio de nombres del paquete
__all__ = ['gui', 'models', 'utils']
