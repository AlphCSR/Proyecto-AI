# 🍷 Predictor de Calidad de Vino

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

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
   git clone https://github.com/AlphCSR/Proyecto-AI.git
   cd proyecto-ai
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate 
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

##  Autores
```

* Cesar Leon 27741713
* Alejandro Seijas 27426702 
* Linda Gruber 29826397
* Nicolas Dias 27488913
* Daniel Cohen 27254276

```