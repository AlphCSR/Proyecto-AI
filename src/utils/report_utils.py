import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from datetime import datetime
from ..config import feature_names, translations, valid_ranges

def generate_feature_importance_plot(model, output_path='assets/feature_importance.png'):
    """
    Genera y guarda el gráfico de importancia de características
    
    Args:
        model: Modelo entrenado (puede ser un pipeline, un estimador directo o un diccionario con 'model' y 'feature_names')
        output_path: Ruta completa donde se guardará el gráfico
        
    Returns:
        str: Ruta del archivo guardado o None si hay un error
    """
    try:
        # Verificar si el modelo es válido
        if model is None:
            print("Error: El modelo no está inicializado")
            return None
            
        # Crear directorio si no existe
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Solo crear directorio si la ruta lo incluye
            os.makedirs(output_dir, exist_ok=True)
        
        # Si el modelo es un diccionario con 'model' y 'feature_names'
        if isinstance(model, dict) and 'model' in model:
            model_obj = model['model']
            if 'feature_names' in model:
                feature_names = model['feature_names']
            else:
                # Intentar obtener los nombres de las características del modelo si es posible
                if hasattr(model_obj, 'feature_importances_'):
                    feature_names = [f'Feature {i+1}' for i in range(len(model_obj.feature_importances_))]
                else:
                    # Si no se puede determinar la longitud, usar nombres genéricos
                    feature_names = [f'Feature {i+1}' for i in range(10)]  # Valor por defecto
            
            # Obtener las importancias si el modelo las tiene
            if hasattr(model_obj, 'feature_importances_'):
                importance = model_obj.feature_importances_
            elif hasattr(model_obj, 'coef_'):
                # Para modelos lineales que usan coef_ en lugar de feature_importances_
                importance = np.abs(model_obj.coef_.flatten())
            else:
                print("Advertencia: No se pudo determinar la importancia de características para el modelo")
                print("El modelo no tiene un atributo 'feature_importances_' o 'coef_' accesible")
                return None
        else:
            # Manejar el caso en que el modelo no es un diccionario
            model_obj = model
            importance = None
            
            # Intentar diferentes formas de obtener las importancias de características
            if hasattr(model_obj, 'feature_importances_'):
                importance = model_obj.feature_importances_
            elif hasattr(model_obj, 'coef_'):
                # Para modelos lineales
                importance = np.abs(model_obj.coef_.flatten())
            elif hasattr(model_obj, 'named_steps') and 'model' in model_obj.named_steps:
                # Si es un pipeline con un paso 'model'
                feature_estimator = model_obj.named_steps['model']
                if hasattr(feature_estimator, 'feature_importances_'):
                    importance = feature_estimator.feature_importances_
                elif hasattr(feature_estimator, 'coef_'):
                    importance = np.abs(feature_estimator.coef_.flatten())
            
            # Si no se pudo obtener las importancias, mostrar mensaje y salir
            if importance is None:
                print("Advertencia: No se pudo determinar la importancia de características para el modelo")
                print("El modelo no tiene un atributo 'feature_importances_' o 'coef_' accesible")
                return None
            
            # Obtener nombres de características
            if hasattr(model_obj, 'feature_names_in_'):
                feature_names = model_obj.feature_names_in_
            elif hasattr(model_obj, 'named_steps') and hasattr(model_obj.named_steps.get('preprocessor'), 'get_feature_names_out'):
                feature_names = model_obj.named_steps['preprocessor'].get_feature_names_out()
            else:
                # Usar nombres de características por defecto o generar nombres genéricos
                try:
                    from ...config import feature_names as default_feature_names
                    feature_names = default_feature_names
                except ImportError:
                    feature_names = [f'Feature {i+1}' for i in range(len(importance))]
        
        # Asegurarse de que feature_names sea una lista
        if not isinstance(feature_names, (list, np.ndarray)):
            feature_names = [f'Feature {i+1}' for i in range(len(importance))]
        
        # Asegurarse de que tenemos la misma cantidad de características
        if len(importance) != len(feature_names):
            print(f"Advertencia: Número de características ({len(feature_names)}) no coincide con la longitud de importancias ({len(importance)})")
            # Usar índices como nombres si no coinciden
            feature_names = [f'Feature {i+1}' for i in range(len(importance))]
        
        # Crear DataFrame para ordenar las características
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Si no hay datos para mostrar, salir
        if len(feature_importance) == 0:
            print("Advertencia: No hay datos de importancia para mostrar")
            return None
        
        # Crear el gráfico
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")
        
        # Limitar a las 15 características más importantes si hay muchas
        max_features = min(15, len(feature_importance))
        plot_data = feature_importance.head(max_features)
        
        # Crear el gráfico de barras
        try:
            ax = sns.barplot(
                x='importance', 
                y='feature', 
                data=plot_data,
                palette='viridis',
                legend=False
            )
            
            # Añadir etiquetas de valor a las barras
            for i, v in enumerate(plot_data['importance']):
                ax.text(v + 0.001, i, f'{v:.3f}', color='black', va='center')
            
            plt.title('Importancia de las Características', fontsize=14, pad=15)
            plt.xlabel('Importancia', fontsize=12)
            plt.ylabel('')
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Gráfico de importancia guardado en: {output_path}")
            return output_path
            
        except Exception as plot_error:
            print(f"Error al generar el gráfico de barras: {str(plot_error)}")
            plt.close()
            return None
        
    except Exception as e:
        print(f"Error al generar el gráfico de importancia de características: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'plt' in locals():
            plt.close()
        return None

def generate_correlation_plot(X, output_path='assets/correlation_matrix.png'):
    """
    Genera y guarda el gráfico de correlación
    
    Args:
        X: DataFrame o array de NumPy con las características
        output_path: Ruta completa donde se guardará el gráfico
    """
    try:
        # Crear directorio si no existe
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Solo crear directorio si la ruta lo incluye
            os.makedirs(output_dir, exist_ok=True)
        
        # Convertir a DataFrame si es un array de NumPy
        if not isinstance(X, pd.DataFrame):
            try:
                # Intentar importar los nombres de características desde la configuración
                from ...config import feature_names
                X_df = pd.DataFrame(X, columns=feature_names)
            except (ImportError, AttributeError):
                # Si no se pueden obtener los nombres, usar índices genéricos
                X_df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
        else:
            X_df = X
        
        # Verificar que haya datos para mostrar
        if X_df.empty or X_df.shape[1] < 2:
            print("Advertencia: No hay suficientes características para generar la matriz de correlación")
            return None
        
        # Calcular matriz de correlación
        corr = X_df.corr()
        
        # Crear máscara para la matriz triangular superior
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Configurar el tamaño del gráfico
        plt.figure(figsize=(12, 10))
        
        # Crear el mapa de calor con estilo mejorado
        sns.set(font_scale=0.9)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        # Crear el heatmap
        ax = sns.heatmap(
            corr, 
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Coeficiente de correlación"},
            annot=True,
            fmt='.2f',
            annot_kws={"size": 9}
        )
        
        # Mejorar la legibilidad
        plt.title('Matriz de Correlación de Características', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar la imagen
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"Error al generar el gráfico de correlación: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_quality_distribution(y, output_path='assets/quality_distribution.png'):
    """
    Genera y guarda el gráfico de distribución de calidades
    
    Args:
        y: Serie, array o lista con las calificaciones de calidad
        output_path: Ruta completa donde se guardará el gráfico
        
    Returns:
        str: Ruta del archivo guardado o None si hay un error
    """
    try:
        # Verificar si y es None o está vacío
        if y is None:
            print("Error: No se proporcionaron datos de calidad")
            return None
            
        # Convertir a pandas Series si es necesario
        if not isinstance(y, pd.Series):
            try:
                y = pd.Series(y)
            except Exception as e:
                print(f"Error al convertir los datos a Series: {str(e)}")
                return None
        
        # Filtrar valores nulos o no numéricos
        y = y.dropna()
        if y.empty:
            print("Error: No hay datos válidos para generar el gráfico de distribución")
            return None
        
        # Redondear los valores de calidad a enteros si son flotantes
        if pd.api.types.is_float_dtype(y):
            y = y.round().astype(int)
        
        # Crear directorio si no existe
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Solo crear directorio si la ruta lo incluye
            os.makedirs(output_dir, exist_ok=True)
        
        # Contar la frecuencia de cada calidad
        quality_counts = y.value_counts().sort_index()
        
        # Si no hay suficientes valores únicos, no generar el gráfico
        if len(quality_counts) < 2:
            print("Advertencia: No hay suficientes valores únicos para el gráfico de distribución")
            return None
        
        # Crear el gráfico con estilo mejorado
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        # Crear el gráfico de barras
        try:
            ax = sns.barplot(
                x=quality_counts.index.astype(str),  # Asegurar que los índices sean strings
                y=quality_counts.values, 
                palette='viridis',
                edgecolor='black',
                linewidth=0.7
            )
            
            # Añadir etiquetas y título
            plt.title('Distribución de Calidades en el Conjunto de Entrenamiento', 
                     fontsize=14, pad=20)
            plt.xlabel('Calidad', fontsize=12)
            plt.ylabel('Cantidad de Muestras', fontsize=12)
            
            # Ajustar el espaciado de las etiquetas del eje X
            plt.xticks(rotation=0 if len(quality_counts) < 10 else 45)
            
            # Añadir valores en las barras con mejor formato
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.annotate(
                    f'{int(height):,}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    xytext=(0, 5),
                    textcoords='offset points'
                )
            
            # Mejorar los ejes
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Ajustar el diseño para evitar cortes
            plt.tight_layout()
            
            # Guardar la imagen
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Gráfico de distribución guardado en: {output_path}")
            return output_path
            
        except Exception as plot_error:
            print(f"Error al generar el gráfico de distribución: {str(plot_error)}")
            import traceback
            traceback.print_exc()
            if 'plt' in locals():
                plt.close()
            return None
            
    except Exception as e:
        print(f"Error inesperado en generate_quality_distribution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_feature_recommendation(feature, value, mean_val, min_val, max_val, input_data=None):
    """Genera recomendaciones específicas para cada característica del vino"""
    try:
        feature_name = translations.get(feature, feature)
        value = float(value)
        mean_val = float(mean_val)
        min_val = float(min_val)
        max_val = float(max_val)
        
        recommendations = []
        
        # Definir umbrales para considerar un valor como alto o bajo
        low_threshold = min_val * 0.9
        high_threshold = max_val * 1.1
        
        # Mapeo de características y sus recomendaciones
        feature_advice = {
            'fixed acidity': {
                'high': 'La acidez fija es alta, lo que puede hacer que el vino sea demasiado ácido. Considera reducirla para lograr un mejor equilibrio.',
                'low': 'La acidez fija es baja, lo que puede hacer que el vino sea plano. Considera aumentarla para mejorar la frescura.'
            },
            'volatile acidity': {
                'high': 'La acidez volátil es alta, lo que puede causar sabores avinagrados. Reducirla mejoraría significativamente la calidad.',
                'low': 'La acidez volátil está en un buen rango, lo que es positivo para el perfil aromático del vino.'
            },
            'citric acid': {
                'high': 'El contenido de ácido cítrico es alto, lo que puede hacer que el vino sea demasiado ácido. Considera reducirlo ligeramente.',
                'low': 'El contenido de ácido cítrico es bajo. Un ligero aumento podría mejorar la frescura y el equilibrio del vino.'
            },
            'residual sugar': {
                'high': 'El azúcar residual es alto, lo que puede hacer que el vino sea demasiado dulce. Considera reducirlo para un perfil más equilibrado.',
                'low': 'El azúcar residual es bajo. Un ligero aumento podría mejorar el equilibrio, especialmente si la acidez es alta.'
            },
            'chlorides': {
                'high': 'El contenido de cloruros es alto, lo que puede dar un sabor salado no deseado. Reducir los cloruros mejoraría el perfil de sabor.',
                'low': 'El contenido de cloruros es adecuado, lo que es positivo para el equilibrio general del vino.'
            },
            'free sulfur dioxide': {
                'high': 'El dióxido de azufre libre es alto. Aunque ayuda a la conservación, niveles excesivos pueden afectar negativamente el sabor.',
                'low': 'El dióxido de azufre libre es bajo. Considera aumentarlo ligeramente para mejorar la estabilidad del vino.'
            },
            'total sulfur dioxide': {
                'high': 'El dióxido de azufre total es alto. Niveles excesivos pueden afectar negativamente el sabor y el aroma.',
                'low': 'El dióxido de azufre total es adecuado para la conservación del vino.'
            },
            'density': {
                'high': 'La densidad es alta, lo que puede indicar un vino con más cuerpo o más dulce. Considera ajustar los sólidos disueltos.',
                'low': 'La densidad es adecuada para el tipo de vino.'
            },
            'pH': {
                'high': 'El pH es alto, lo que puede afectar la estabilidad microbiológica. Considera reducirlo ligeramente.',
                'low': 'El pH es bajo, lo que puede hacer que el vino sea demasiado ácido. Considera aumentarlo ligeramente.'
            },
            'sulphates': {
                'high': 'El contenido de sulfatos es alto, lo que puede afectar negativamente el sabor. Considera reducirlo.',
                'low': 'El contenido de sulfatos es bajo. Un ligero aumento podría mejorar la estabilidad y el carácter del vino.'
            },
            'alcohol': {
                'high': 'El contenido de alcohol es alto, lo que puede hacer que el vino sea demasiado cálido. Considera reducirlo para un mejor equilibrio.',
                'low': 'El contenido de alcohol es bajo. Un ligero aumento podría mejorar el cuerpo y la sensación en boca.'
            }
        }
        
        # Generar recomendación basada en los valores
        if feature in feature_advice:
            if value > high_threshold:
                recommendations.append(feature_advice[feature]['high'])
            elif value < low_threshold:
                recommendations.append(feature_advice[feature]['low'])
        
        # Añadir recomendaciones específicas basadas en combinaciones de características
        if input_data is not None:
            if (feature == 'volatile acidity' and value > high_threshold and 
                'alcohol' in input_data and float(input_data.get('alcohol', 0)) > 12.5):
                recommendations.append("La combinación de alta acidez volátil y alto contenido de alcohol puede ser problemática. Considera reducir ambos parámetros para mejorar el equilibrio.")
            
            if (feature == 'residual sugar' and value > high_threshold and 
                'total sulfur dioxide' in input_data and float(input_data.get('total sulfur dioxide', 0)) > 150):
                recommendations.append("El alto contenido de azúcar residual junto con altos niveles de dióxido de azufre puede afectar negativamente el sabor. Considera reducir ambos parámetros.")
        
        return recommendations
    except Exception as e:
        print(f"Error al generar recomendaciones para {feature}: {str(e)}")
        return []

def generate_prediction_report(input_data, prediction, X_train, y_train, output_dir='reports'):
    """Genera un reporte completo de la predicción con recomendaciones detalladas"""
    try:
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Asegurarse de que X_train es un DataFrame
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        
        # Asegurar que los nombres de las columnas sean strings
        X_train.columns = X_train.columns.astype(str)
        
        # Obtener estadísticas básicas
        stats = X_train.describe()
        
        # Crear el contenido del reporte
        report = []
        report.append("# Reporte de Predicción de Calidad de Vino\n")
        
        # Agregar información de la predicción
        report.append(f"## Predicción de Calidad")
        report.append(f"- **Puntuación predicha:** {prediction:.2f}")
        report.append(f"- **Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Agregar comparación con valores de entrenamiento
        report.append("## Comparación con el Conjunto de Entrenamiento")
        report.append("| Característica | Valor Ingresado | Media (Entrenamiento) | Rango Típico | Estado |")
        report.append("|----------------|-----------------|----------------------|--------------|--------|")
        
        # Asegurarse de que input_data es un diccionario
        if not isinstance(input_data, dict):
            if isinstance(input_data, (pd.Series, pd.DataFrame)):
                input_data = input_data.to_dict()
            else:
                input_data = dict(zip(feature_names, input_data))
        
        # Lista para almacenar todas las recomendaciones
        all_recommendations = []
        
        for feature in feature_names:
            feature_str = str(feature)  # Asegurar que el nombre de la característica sea string
            if feature_str in input_data:
                value = input_data[feature_str]
                mean_val = stats.loc['mean', feature_str]
                min_val = stats.loc['25%', feature_str]
                max_val = stats.loc['75%', feature_str]
                
                # Determinar el estado
                status = "✅ Normal"
                if float(value) > float(max_val) * 1.1:
                    status = "⚠️ Alto"
                elif float(value) < float(min_val) * 0.9:
                    status = "⚠️ Bajo"
                
                report.append(
                    f"| {translations.get(feature_str, feature_str)} | {float(value):.2f} | {float(mean_val):.2f} | "
                    f"({float(min_val):.2f} - {float(max_val):.2f}) | {status} |"
                )
                
                # Obtener recomendaciones para esta característica
                feature_recs = get_feature_recommendation(
                    feature=feature_str,
                    value=value,
                    mean_val=mean_val,
                    min_val=min_val,
                    max_val=max_val,
                    input_data=input_data
                )
                # Agregar guiones a cada recomendación
                all_recommendations.extend([f"- {rec}" for rec in feature_recs])
        
        # Agregar recomendaciones basadas en la predicción
        report.append("\n## Recomendaciones Generales")
        
        if prediction >= 7:
            report.append("- **Excelente calidad de vino.** Este es un vino de alta gama con características sobresalientes.")
        elif prediction >= 5:
            report.append("- **Buena calidad de vino.** El vino es equilibrado y agradable, con margen para mejoras menores.")
        else:
            report.append("- **Calidad regular.** El vino podría mejorar significativamente con ajustes en su composición.")
        
        # Agregar recomendaciones específicas si las hay
        if all_recommendations:
            report.append("\n## Recomendaciones Específicas")
            report.append("A continuación se presentan recomendaciones para mejorar la calidad del vino basadas en sus características actuales:")
            report.extend(all_recommendations)
        else:
            report.append("\nEl vino tiene un perfil equilibrado. No se requieren ajustes significativos.")
        
        # Agregar consejos generales de mejora
        report.append("\n## Consejos Generales para Mejorar la Calidad")
        report.append("1. **Equilibrio es clave**: Busca un equilibrio entre acidez, dulzor, alcohol y taninos.")
        report.append("2. **Control de temperaturas**: Asegúrate de que la fermentación se realice a temperaturas adecuadas.")
        report.append("3. **Manejo de barricas**: Considera el uso de barricas de roble para añadir complejidad.")
        report.append("4. **Tiempo de guarda**: Algunos vinos mejoran significativamente con el añejamiento.")
        report.append("5. **Pruebas de laboratorio**: Realiza análisis químicos regulares para monitorear los parámetros clave.")
        
        # Guardar el reporte en un archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f'report_{timestamp}.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return report_path
    except Exception as e:
        print(f"Error al generar el reporte: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
