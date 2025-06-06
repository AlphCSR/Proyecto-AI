# Configuración de la aplicación
import os

APP_TITLE = "🍷 Wine Quality Predictor"
PRIMARY_COLOR = "#722F37"  # Rojo vino
SECONDARY_COLOR = "#FFFAFA"  # Blanco Nieve
SECONDARY_LIGHT = "#F5F5F5"  # Gris muy claro para fondos
ACCENT_COLOR = "#8B0000"  # Rojo oscuro
TEXT_COLOR = "#2C3E50"
WHITE = "#FFFFFF"

# Configuración de modelos y datos
WINE_TYPES = {
    'white': {
        'model_path': os.path.join('src', 'regressors', 'rf_regressor_white.joblib'),
        'data_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    },
    'red': {
        'model_path': os.path.join('src', 'regressors', 'rf_regressor_red.joblib'),
        'data_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    }
}

# Configuración por defecto
DEFAULT_WINE_TYPE = 'white'
MODEL_PATH = WINE_TYPES[DEFAULT_WINE_TYPE]['model_path']
DATA_URL = WINE_TYPES[DEFAULT_WINE_TYPE]['data_url']

# Traducciones de características
translations = {
    'fixed acidity': 'Acidez fija',
    'volatile acidity': 'Acidez volátil',
    'citric acid': 'Ácido cítrico',
    'residual sugar': 'Azúcar residual',
    'chlorides': 'Cloruros',
    'free sulfur dioxide': 'Dióxido de azufre libre',
    'total sulfur dioxide': 'Dióxido de azufre total',
    'density': 'Densidad',
    'pH': 'pH',
    'sulphates': 'Sulfatos',
    'alcohol': 'Alcohol'
}

# Rangos válidos para cada característica
valid_ranges = {
    'fixed acidity': (3.8, 15.9),
    'volatile acidity': (0.08, 1.58),
    'citric acid': (0.0, 1.66),
    'residual sugar': (0.6, 65.8),
    'chlorides': (0.009, 0.611),
    'free sulfur dioxide': (2.0, 289.0),
    'total sulfur dioxide': (9.0, 440.0),
    'density': (0.98711, 1.03898),
    'pH': (2.72, 3.82),
    'sulphates': (0.22, 2.0),
    'alcohol': (8.0, 14.9)
}

# Nombres de las características
feature_names = list(translations.keys())
