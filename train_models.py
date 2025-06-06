import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
import joblib
import time

# Configuración de la barra de progreso
TQDM_AVAILABLE = False
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Si tqdm no está disponible, usamos una implementación mínima
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iter(iterable) if iterable is not None else None
            self.desc = desc
            self.total = total
            self.n = 0
            
        def __iter__(self):
            if self.iterable is None:
                return self
            for item in self.iterable:
                yield item
                self.n += 1
                
        def update(self, n=1):
            self.n += n
            
        def close(self):
            pass
            
        def set_description(self, desc=None):
            self.desc = desc

class ProgressCallback:
    """Clase para manejar el progreso y las métricas"""
    def __init__(self, total_steps=100, callback=None):
        self.total_steps = total_steps
        self.current_step = 0
        self.callback = callback
    
    def update(self, step=None, message=None, metrics=None):
        """Actualiza el progreso"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = min(100, int((self.current_step / self.total_steps) * 100))
        
        if self.callback:
            self.callback(progress, message, metrics)

# Asegurarse de que el directorio de trabajo sea correcto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Crear directorio de modelos si no existe
os.makedirs('src/regressors', exist_ok=True)

def train_wine_model(url, model_name, progress_callback=None):
    """
    Entrena un modelo de regresión para predecir la calidad del vino
    
    Args:
        url (str): URL del dataset a utilizar
        model_name (str): Nombre del modelo (se usará para guardar el archivo)
        progress_callback (callable, opcional): Función para reportar el progreso
        
    Returns:
        dict: Diccionario con las métricas del modelo entrenado
    """
    try:
        # Inicializar el callback de progreso si no se proporciona
        if progress_callback is None:
            progress_callback = lambda p, m=None, metrics=None: None
        
        # Configurar pasos totales para la barra de progreso
        total_steps = 100
        progress = ProgressCallback(total_steps=total_steps, callback=progress_callback)
        
        # Paso 1: Cargar datos
        progress.update(message="Cargando datos...")
        df = pd.read_csv(url, sep=";")
        
        # Preparar características y objetivo
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Dividir datos
        progress.update(message="Dividiendo datos en conjuntos de entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Crear pipeline del modelo
        progress.update(message="Configurando el modelo...")
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('regressor', RandomForestRegressor(
                n_estimators=50,  # Reducir el número de árboles para acelerar el entrenamiento
                random_state=42,
                n_jobs=1,  # Desactivar paralelización para evitar problemas en el ejecutable
                verbose=0
            ))
        ])
        
        # Entrenar modelo con barra de progreso
        progress.update(message="Entrenando modelo...")
        
        # Configurar barra de progreso para el entrenamiento
        n_estimators = model.named_steps['regressor'].n_estimators
        
        # Crear un wrapper para el callback de progreso
        def fit_with_progress(estimator, X, y):
            # Guardar el método fit original
            original_fit = estimator.fit
            
            # Si es un modelo con n_estimators, entrenar árbol por árbol
            if hasattr(estimator, 'n_estimators'):
                # Configurar la barra de progreso si está disponible
                if TQDM_AVAILABLE:
                    pbar = tqdm(total=n_estimators, desc="Árboles entrenados")
                
                # Entrenar árbol por árbol
                for i in range(n_estimators):
                    # Configurar el número de árboles a entrenar
                    estimator.n_estimators = i + 1
                    
                    # Entrenar el modelo con los árboles actuales
                    if hasattr(estimator, 'warm_start') and estimator.warm_start:
                        if i == 0:
                            original_fit(X, y)
                        else:
                            original_fit(X, y, xgb_model=estimator)
                    else:
                        # Para RandomForest, necesitamos crear un nuevo estimador cada vez
                        if i == 0:
                            # En la primera iteración, usar el estimador original
                            original_fit(X, y)
                        else:
                            # En iteraciones posteriores, crear un nuevo estimador
                            new_estimator = clone(estimator)
                            new_estimator.n_estimators = i + 1
                            new_estimator.fit(X, y)
                            # Copiar los árboles entrenados al estimador original
                            if hasattr(estimator, 'estimators_'):
                                estimator.estimators_ = new_estimator.estimators_
                            if hasattr(estimator, 'n_features_in_'):
                                estimator.n_features_in_ = new_estimator.n_features_in_
                            if hasattr(estimator, 'feature_importances_'):
                                estimator.feature_importances_ = new_estimator.feature_importances_
                    
                    # Actualizar progreso
                    current_progress = 60 + int((i + 1) / n_estimators * 30)  # 60-90%
                    progress.update(
                        step=current_progress,
                        message=f"Entrenando árbol {i+1}/{n_estimators}",
                        metrics={
                            'epoch': f"{i+1}/{n_estimators}",
                            'status': f"Entrenando árbol {i+1}/{n_estimators}"
                        }
                    )
                    
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                
                if TQDM_AVAILABLE:
                    pbar.close()
            else:
                # Para otros estimadores sin n_estimators, entrenar normalmente
                original_fit(X, y)
        
        # Entrenar el modelo con el wrapper de progreso
        fit_with_progress(model, X_train, y_train)
        
        # Evaluar modelo
        progress.update(step=90, message="Evaluando modelo...")
        
        # Calcular métricas
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2 = r2_score(y_test, y_test_pred)
        
        # Validación cruzada con un solo job para evitar problemas de paralelización
        cv_scores = cross_val_score(
            model, X, y, cv=3,  # Reducir cv para acelerar
            scoring='r2',
            n_jobs=1  # Desactivar paralelización para evitar problemas
        )
        
        # Mostrar métricas
        metrics = {
            'train_score': f"{train_score:.4f}",
            'test_score': f"{test_score:.4f}",
            'rmse': f"{rmse:.4f}",
            'r2': f"{r2:.4f}",
            'cv_mean': f"{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}",
            'status': 'Evaluación completada'
        }
        progress.update(step=95, message="Evaluación completada", metrics=metrics)
        
        # Guardar modelo
        progress.update(step=97, message="Guardando modelo...")
        model_path = os.path.join('src', 'regressors', f'rf_regressor_{model_name}.joblib')
        joblib.dump(model, model_path)
        
        # Completar progreso
        progress.update(step=100, message="¡Entrenamiento completado!")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'rmse': rmse,
            'r2': r2,
            'cv_scores': cv_scores,
            'model_path': os.path.abspath(model_path)
        }
        
    except Exception as e:
        error_msg = f"Error entrenando modelo {model_name}: {str(e)}"
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}", {'status': 'Error en el entrenamiento'})
        raise RuntimeError(error_msg) from e

# URLs for the datasets
DATASETS = {
    'white': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    'red': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
}

if __name__ == "__main__":
    # Solo ejecutar si se llama directamente al script
    print("Iniciando entrenamiento de modelos...")
    
    # Función de callback simple para la línea de comandos
    def print_progress(progress, message=None, metrics=None):
        if message:
            print(f"\r{message} - {progress}%", end='')
        if progress == 100:
            print("\n¡Completado!")
    
    # Entrenar modelos para ambos tipos de vino
    for wine_type, url in DATASETS.items():
        print(f"\nEntrenando modelo para vino {wine_type}...")
        try:
            result = train_wine_model(url, wine_type, progress_callback=print_progress)
            print(f"\nResultados para {wine_type}:")
            print(f"  - R² (train): {result['train_score']:.4f}")
            print(f"  - R² (test):  {result['test_score']:.4f}")
            print(f"  - RMSE:       {result['rmse']:.4f}")
            print(f"  - R²:         {result['r2']:.4f}")
            print(f"  - CV R²:      {np.mean(result['cv_scores']):.4f} ± {np.std(result['cv_scores']):.4f}")
            print(f"  - Modelo guardado en: {result['model_path']}")
        except Exception as e:
            print(f"\nError durante el entrenamiento del modelo {wine_type}: {str(e)}")
    
    print("\n¡Entrenamiento de modelos completado!")
