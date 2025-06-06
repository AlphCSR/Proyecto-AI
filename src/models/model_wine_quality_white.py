# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset desde UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Cargar los datasets
df = pd.read_csv(url, sep=";")

# Verificar valores ausentes (aunque el dataset original no los tiene)
missing_values = df.isnull().sum()
print(f"Valores ausentes en las columnas:\n{missing_values[missing_values > 0]}")

# Exploración rápida
print("\nMuestra de los primeros 5 registros:")
print(df.head())

# Gráfico de distribución de la variable objetivo (quality)
plt.figure(figsize=(12, 6))
sns.histplot(df['quality'], bins=15, kde=True, color='blue', alpha=0.6)
plt.title("Distribución de Calidad del Vino")
plt.xlabel('Calidad')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.savefig('assets/distribution_white.png')
plt.close()

# Preparar los datos para el modelo
X = df.drop('quality', axis=1)  # Características
y = df['quality']  # Etiqueta de calidad

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definición del pipeline de imputación y modelo
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Imputación por mediana
    ('model', RandomForestRegressor(n_estimators=300, random_state=42))  # Modelo RandomForest
])

# Entrenar el modelo
pipe.fit(X_train, y_train)

# Gráfico de la matriz de correlación entre las características
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación de las Características del Vino")
plt.tight_layout()
plt.savefig('assets/correlation_matrix_white.png')
plt.close()

# Gráfico de la importancia de las características para el modelo
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")
feature_importances = pipe.named_steps['model'].feature_importances_
features = X.columns
sns.barplot(x=features, y=feature_importances, palette="Blues_d")
plt.title("Importancia de las Características para el Modelo")
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/feature_importance_white.png')
plt.close()

# Evaluar el modelo en el conjunto de prueba
preds = pipe.predict(X_test)
rmse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\n**RMSE**: {rmse:.3f}")
print(f"**R²**: {r2:.3f}")

# Guardar el modelo entrenado
joblib.dump(pipe, "src/regressors/rf_regressor_white.joblib")
print("\nModelo guardado como 'rf_regressor_white.joblib'")
