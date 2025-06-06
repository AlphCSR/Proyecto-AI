import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

from src.config import DATA_URL, feature_names

def load_training_data(data_url=None):
    """
    Carga los datos de entrenamiento desde la URL especificada
    
    Args:
        data_url: URL del archivo CSV con los datos. Si es None, usa la URL por defecto.
    
    Returns:
        Tuple (X, y) con características y variable objetivo
    """
    try:
        # Usar la URL proporcionada o la URL por defecto
        url = data_url if data_url is not None else DATA_URL
        
        # Cargar datos
        df = pd.read_csv(url, sep=";")
        
        # Verificar que las columnas necesarias existan
        if 'quality' not in df.columns:
            raise ValueError("La columna 'quality' no se encuentra en los datos")
        
        # Separar características (X) y variable objetivo (y)
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Verificar que no haya valores faltantes
        if X.isnull().any().any() or y.isnull().any():
            print("Advertencia: Se encontraron valores faltantes en los datos")
        
        return X, y
        
    except pd.errors.EmptyDataError:
        raise Exception("El archivo de datos está vacío")
    except pd.errors.ParserError:
        raise Exception("Error al analizar el archivo CSV. Verifique el formato.")
    except Exception as e:
        raise Exception(f"Error al cargar los datos de entrenamiento: {str(e)}")

def calculate_statistics(X, target=None):
    """
    Calcula estadísticas descriptivas para las características
    
    Args:
        X: DataFrame con las características
        target: Serie con la variable objetivo (opcional)
        
    Returns:
        dict: Diccionario con estadísticas básicas
    """
    try:
        # Calcular estadísticas básicas
        stats = {
            'count': len(X),
            'mean': float(X.mean().mean()) if not X.empty else 0,
            'min': float(X.min().min()) if not X.empty else 0,
            'max': float(X.max().max()) if not X.empty else 0,
            'features': {}
        }
        
        # Si hay una variable objetivo, agregar sus estadísticas
        if target is not None and not target.empty:
            stats.update({
                'target_mean': float(target.mean()),
                'target_median': float(target.median()),
                'target_min': float(target.min()),
                'target_max': float(target.max()),
                'target_std': float(target.std())
            })
        
        return stats
        
    except Exception as e:
        print(f"Error al calcular estadísticas: {str(e)}")
        # Devolver estadísticas vacías en caso de error
        return {
            'count': 0,
            'mean': 0,
            'min': 0,
            'max': 0,
            'features': {}
        }

def load_model(model_path=None):
    """
    Carga el modelo entrenado desde la ruta especificada
    
    Args:
        model_path: Ruta al archivo del modelo. Si es None, usa la ruta por defecto.
    
    Returns:
        dict: Diccionario con el modelo y los nombres de las características
    """
    try:
        # Usar la ruta proporcionada o la ruta por defecto
        if model_path is None:
            # Si no se proporciona una ruta, usar el modelo blanco por defecto
            path = os.path.join('src', 'regressors', 'rf_regressor_white.joblib')
        else:
            path = model_path
        
        # Verificar si el archivo existe
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {path}")
        
        # Cargar el modelo
        model_data = joblib.load(path)
        
        # Verificar si el modelo ya está en el formato esperado (diccionario con 'model' y 'feature_names')
        if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
            model = model_data['model']
            feature_names = model_data['feature_names']
        else:
            # Si no está en el formato esperado, asumir que es el modelo directamente
            model = model_data
            feature_names = [
                'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol'
            ]
        
        # Verificar que el modelo tenga el método predict
        if not hasattr(model, 'predict'):
            raise ValueError("El modelo cargado no tiene el método 'predict'")
        
        # Devolver un diccionario con el modelo y los nombres de las características
        return {
            'model': model,
            'feature_names': feature_names
        }
        
    except FileNotFoundError as e:
        raise Exception(f"Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {str(e)}")

def quality_label(score):
    """
    Devuelve la etiqueta de calidad basada en la puntuación
    
    Args:
        score: Puntuación de calidad (puede ser numpy.float64, float o int)
        
    Returns:
        tuple: (etiqueta, color_hex)
    """
    try:
        # Asegurarse de que el score sea un tipo numérico estándar
        score = float(score) if hasattr(score, 'item') else float(score)
        
        if score >= 7:
            return 'Excelente', '#2ecc71'  # Verde
        elif score >= 5:
            return 'Bueno', '#3498db'      # Azul
        else:
            return 'Regular', '#e74c3c'    # Rojo
    except (ValueError, TypeError) as e:
        print(f"Advertencia: No se pudo determinar la etiqueta de calidad para el valor {score}: {str(e)}")
        return 'Desconocido', '#95a5a6'  # Gris para valores desconocidos
