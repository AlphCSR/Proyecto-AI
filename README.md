# 🍷 Predictor de Calidad de Vino

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema de predicción de calidad de vino que utiliza técnicas avanzadas de Machine Learning para predecir la calidad del vino basado en sus características físico-químicas.

## 📋 Descripción

Este proyecto implementa modelos de Machine Learning para predecir la calidad del vino en una escala del 0 al 10, utilizando características como acidez, azúcar residual, pH, entre otros. El sistema incluye una interfaz gráfica intuitiva para realizar predicciones y analizar resultados.

## 🚀 Características

- Predicción de calidad de vino en escala del 0 al 10
- Interfaz gráfica amigable
- Múltiples algoritmos de Machine Learning implementados
- Análisis exploratorio de datos incluido
- Capacidad de entrenar nuevos modelos
- Documentación completa

## 📦 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Dependencias listadas en `requirements.txt`

## 🛠️ Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/proyecto-ia-vino.git
   cd proyecto-ia-vino
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # En Windows
   # O en Linux/Mac: source .venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Uso

Para ejecutar la aplicación:

```bash
python -m src.main
```

### Entrenamiento de modelos

Para entrenar nuevos modelos:

```bash
python -m train_models
```

## 🗂️ Estructura del Proyecto

```
.
├── data/                    # Conjuntos de datos
│   ├── raw/                 # Datos sin procesar
│   └── processed/           # Datos procesados
├── reports/                 # Reportes y análisis
├── src/                     # Código fuente
│   ├── gui/                 # Interfaz gráfica
│   ├── models/              # Modelos de ML
│   ├── regressors/          # Regresores personalizados
│   ├── utils/               # Utilidades
│   ├── config.py            # Configuración
│   └── main.py              # Punto de entrada
├── .venv/                   # Entorno virtual
├── requirements.txt         # Dependencias
├── train_models.py          # Script de entrenamiento
└── README.md               # Este archivo
```

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor, lee nuestras [pautas de contribución](CONTRIBUTING.md) antes de enviar un pull request.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más información.

## 🙏 Agradecimientos

- Conjunto de datos: [UCI Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- A todos los contribuyentes que han ayudado a mejorar este proyecto.
