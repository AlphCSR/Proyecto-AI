import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')  # Usar el backend de Tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64

class ReportScreen:
    def __init__(self, parent, model, X_train, y_train):
        self.parent = parent
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.current_report_data = None
        
        # Crear el frame principal
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill='both', expand=True)  # Hacer que el frame llene todo el espacio
        
        # Configurar el evento de la rueda del ratón
        self.main_frame.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Canvas con scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(
            self.main_frame, 
            orient="vertical", 
            command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configurar el canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        # Crear la ventana del canvas con el frame desplazable
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", tags=("inner_frame",))
        
        # Configurar la barra de desplazamiento
        self.scrollbar.config(command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Habilitar el desplazamiento con la rueda del ratón en el canvas
        self.canvas.bind("<Enter>", lambda _: self.canvas.focus_set())
        self.canvas.bind("<Leave>", lambda _: self.canvas.master.focus_set())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Configurar el grid del frame desplazable
        self.scrollable_frame.columnconfigure(0, weight=1)
        self.scrollable_frame.rowconfigure(0, weight=1)
        
        # Configurar el grid para el redimensionamiento
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Empaquetar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configurar el frame para que se expanda con la ventana
        self.scrollable_frame.columnconfigure(0, weight=1)
        self.scrollable_frame.rowconfigure(0, weight=1)
        
        # Hacer que el canvas se redimensione con la ventana
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Frame para el contenido
        self.content_frame = ttk.Frame(self.scrollable_frame, padding=20)
        self.content_frame.pack(fill="both", expand=True)
        
        # Configurar el grid del frame de contenido
        self.content_frame.columnconfigure(0, weight=1)
        
        # Mostrar mensaje inicial
        self.show_initial_message()
    
    def show_initial_message(self):
        """Muestra un mensaje inicial cuando no hay reporte"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        ttk.Label(
            self.content_frame,
            text="No hay datos de reporte disponibles.",
            font=('Segoe UI', 12)
        ).pack(pady=50)
        
        ttk.Label(
            self.content_frame,
            text="Realiza una predicción y genera un reporte para ver los detalles aquí.",
            font=('Segoe UI', 10)
        ).pack()
    
    def show_report(self, input_data, prediction):
        """Muestra el reporte detallado"""
        self.current_report_data = (input_data, prediction)
        
        # Limpiar el frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Título del reporte
        ttk.Label(
            self.content_frame,
            text="Reporte Detallado de Calidad de Vino",
            font=('Segoe UI', 16, 'bold'),
            foreground='#722F37'
        ).pack(pady=(0, 20), anchor='w')
        
        # Sección de resumen
        self.add_section("Resumen de la Predicción")
        self.add_prediction_summary(prediction)
        
        # Sección de comparación con valores típicos
        self.add_section("Comparación con Valores Típicos")
        self.add_comparison_table(input_data)
        
        # Sección de recomendaciones
        self.add_section("Recomendaciones de Mejora")
        self.add_recommendations(input_data, prediction)
        
        # Sección de gráficos (deshabilitada)
        # self.add_section("Análisis Gráfico")
        # self.add_charts()
    
    def add_section(self, title):
        """Agrega un título de sección"""
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill='x', pady=(20, 10), padx=10)
        
        ttk.Label(
            frame,
            text=title,
            font=('Segoe UI', 12, 'bold'),
            foreground='#2C3E50'
        ).pack(anchor='w')
        
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=5)
    
    def add_prediction_summary(self, prediction):
        """Agrega el resumen de la predicción"""
        category, color = self.get_quality_label(prediction)
        
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill='x', padx=20, pady=10)
        
        # Puntuación
        ttk.Label(
            frame,
            text=f"Puntuación de Calidad: {prediction:.2f}/10",
            font=('Segoe UI', 14, 'bold'),
            foreground=color
        ).pack(anchor='w')
        
        # Categoría
        ttk.Label(
            frame,
            text=f"Categoría: {category}",
            font=('Segoe UI', 12),
            foreground=color
        ).pack(anchor='w', pady=(5, 10))
        
        # Descripción
        if prediction >= 7:
            desc = "¡Excelente! Este vino tiene características sobresalientes que lo hacen excepcional."
        elif prediction >= 5:
            desc = "Buena calidad. Este vino cumple con los estándares de calidad esperados."
        else:
            desc = "Calidad regular. Hay margen de mejora en varias características."
            
        ttk.Label(
            frame,
            text=desc,
            font=('Segoe UI', 10),
            wraplength=700,
            justify='left'
        ).pack(anchor='w')
    
    def add_comparison_table(self, input_data):
        """Agrega una tabla comparativa con los valores típicos"""
        try:
            # Crear el frame para la tabla
            table_frame = ttk.Frame(self.content_frame)
            table_frame.pack(fill='x', padx=20, pady=10)
            
            # Encabezados
            headers = ['Característica', 'Tu Valor', 'Rango Típico', 'Estado']
            for col, header in enumerate(headers):
                ttk.Label(
                    table_frame,
                    text=header,
                    font=('Segoe UI', 9, 'bold'),
                    padding=5,
                    relief='ridge'
                ).grid(row=0, column=col, sticky='nsew')
            
            # Asegurarse de que X_train es un DataFrame
            if not hasattr(self, 'X_train') or self.X_train is None:
                raise ValueError("No se encontraron datos de entrenamiento")
                
            # Asegurarse de que las columnas sean strings
            self.X_train.columns = self.X_train.columns.astype(str)
            
            # Datos
            row = 1  # Empezar desde la fila 1 (0 es el encabezado)
            for feature, value in input_data.items():
                try:
                    # Verificar si la característica existe en los datos de entrenamiento
                    if feature not in self.X_train.columns:
                        print(f"Advertencia: La característica '{feature}' no se encontró en los datos de entrenamiento")
                        continue
                        
                    # Obtener estadísticas
                    stats = self.X_train[feature].describe()
                    
                    # Determinar el estado
                    if pd.isna(value):
                        status = "No especificado"
                        status_color = "#666666"
                    elif '25%' in stats and '75%' in stats and stats['25%'] <= value <= stats['75%']:
                        status = "Óptimo"
                        status_color = "#27ae60"
                    elif '10%' in stats and '90%' in stats and stats['10%'] <= value <= stats['90%']:
                        status = "Aceptable"
                        status_color = "#f39c12"
                    else:
                        status = "Fuera de rango"
                        status_color = "#e74c3c"
                    
                    # Nombre de la característica traducido
                    feature_name = feature
                    if hasattr(self, 'translations') and feature in self.translations:
                        feature_name = self.translations[feature]
                    
                    # Fila de datos
                    ttk.Label(
                        table_frame,
                        text=feature_name,
                        padding=5,
                        relief='ridge'
                    ).grid(row=row, column=0, sticky='nsew')
                    
                    ttk.Label(
                        table_frame,
                        text=f"{float(value):.2f}" if not pd.isna(value) else "N/A",
                        padding=5,
                        relief='ridge'
                    ).grid(row=row, column=1, sticky='nsew')
                    
                    # Mostrar rango típico si está disponible
                    range_text = "N/A"
                    if '25%' in stats and '75%' in stats:
                        range_text = f"({stats['25%']:.2f} - {stats['75%']:.2f})"
                        
                    ttk.Label(
                        table_frame,
                        text=range_text,
                        padding=5,
                        relief='ridge'
                    ).grid(row=row, column=2, sticky='nsew')
                    
                    ttk.Label(
                        table_frame,
                        text=status,
                        foreground=status_color,
                        padding=5,
                        relief='ridge'
                    ).grid(row=row, column=3, sticky='nsew')
                    
                    row += 1
                    
                except Exception as e:
                    print(f"Error al procesar la característica '{feature}': {str(e)}")
                    continue
            
            # Configurar el grid
            for i in range(4):
                table_frame.columnconfigure(i, weight=1)
                
        except Exception as e:
            print(f"Error al crear la tabla comparativa: {str(e)}")
            # Mostrar mensaje de error en la interfaz
            ttk.Label(
                self.content_frame,
                text=f"Error al generar la tabla comparativa: {str(e)}",
                foreground='red',
                wraplength=700
            ).pack(pady=10)
    
    def add_recommendations(self, input_data, prediction):
        """Agrega recomendaciones de mejora utilizando la función get_feature_recommendation"""
        try:
            # Importar la función de recomendación
            from src.utils.report_utils import get_feature_recommendation, translations
            
            frame = ttk.Frame(self.content_frame)
            frame.pack(fill='x', padx=20, pady=10)
            
            # Sección de recomendaciones generales
            ttk.Label(
                frame,
                text="Recomendaciones Generales",
                font=('Segoe UI', 10, 'bold'),
                foreground='#2C3E50'
            ).pack(anchor='w', pady=(0, 10))
            
            # Añadir recomendación general basada en la predicción
            if prediction >= 7:
                general_text = "• Excelente calidad de vino. "
                detail_text = "Este es un vino de alta gama con características sobresalientes."
                color = "#27ae60"
            elif prediction >= 5:
                general_text = "• Buena calidad de vino. "
                detail_text = "El vino es equilibrado y agradable, con margen para mejoras menores."
                color = "#f39c12"
            else:
                general_text = "• Calidad regular. "
                detail_text = "El vino podría mejorar significativamente con ajustes en su composición."
                color = "#e74c3c"
            
            # Frame para la recomendación general
            rec_frame = ttk.Frame(frame)
            rec_frame.pack(fill='x', pady=(0, 15), anchor='w')
            
            # Texto en negrita
            ttk.Label(
                rec_frame,
                text=general_text,
                font=('Segoe UI', 10, 'bold'),
                foreground=color
            ).pack(side='left')
            
            # Texto normal
            ttk.Label(
                rec_frame,
                text=detail_text,
                font=('Segoe UI', 10),
                foreground=color
            ).pack(side='left')
            
            # Sección de recomendaciones específicas
            ttk.Label(
                frame,
                text="Recomendaciones Específicas",
                font=('Segoe UI', 10, 'bold'),
                foreground='#2C3E50'
            ).pack(anchor='w', pady=(10, 5))
            
            # Obtener estadísticas para cada característica
            stats = self.X_train.describe()
            
            # Lista para almacenar todas las recomendaciones
            all_recommendations = []
            
            # Generar recomendaciones para cada característica
            for feature, value in input_data.items():
                try:
                    if feature not in self.X_train.columns or pd.isna(value):
                        continue
                        
                    # Obtener estadísticas para esta característica
                    feature_stats = stats[feature]
                    mean_val = feature_stats.get('mean', 0)
                    min_val = feature_stats.get('25%', 0)  # Usamos el percentil 25 como mínimo
                    max_val = feature_stats.get('75%', 0)   # Usamos el percentil 75 como máximo
                    
                    # Obtener recomendaciones para esta característica
                    feature_recs = get_feature_recommendation(
                        feature=feature,
                        value=value,
                        mean_val=mean_val,
                        min_val=min_val,
                        max_val=max_val,
                        input_data=input_data
                    )
                    
                    # Traducir el nombre de la característica
                    feature_name = translations.get(feature, feature)
                    
                    # Formatear recomendaciones
                    if feature_recs:
                        all_recommendations.append((feature_name, feature_recs))
                
                except Exception as e:
                    print(f"Error al generar recomendación para {feature}: {str(e)}")
                    continue
            
            # Mostrar recomendaciones específicas
            if all_recommendations:
                for feature_name, recs in all_recommendations:
                    # Frame para cada recomendación
                    rec_frame = ttk.Frame(frame)
                    rec_frame.pack(fill='x', pady=(0, 5), anchor='w')
                    
                    # Punto de viñeta
                    ttk.Label(
                        rec_frame,
                        text="• ",
                        font=('Segoe UI', 10)
                    ).pack(side='left')
                    
                    # Nombre de la característica en negrita
                    ttk.Label(
                        rec_frame,
                        text=f"{feature_name}: ",
                        font=('Segoe UI', 10, 'bold')
                    ).pack(side='left')
                    
                    # Recomendaciones
                    ttk.Label(
                        rec_frame,
                        text=", ".join(recs),
                        font=('Segoe UI', 10)
                    ).pack(side='left')
            else:
                ttk.Label(
                    frame,
                    text="El vino tiene un perfil equilibrado. No se requieren ajustes significativos.",
                    wraplength=700,
                    justify='left',
                    padding=(0, 5, 0, 5),
                    font=('Segoe UI', 10, 'italic')
                ).pack(anchor='w')
            
            # Sección de consejos generales
            ttk.Label(
                frame,
                text="Consejos Generales para Mejorar la Calidad",
                font=('Segoe UI', 10, 'bold'),
                foreground='#2C3E50'
            ).pack(anchor='w', pady=(20, 5))
            
            consejos = [
                ("1. Equilibrio es clave", "Busca un equilibrio entre acidez, dulzor, alcohol y taninos."),
                ("2. Control de temperaturas", "Asegúrate de que la fermentación se realice a temperaturas adecuadas."),
                ("3. Manejo de barricas", "Considera el uso de barricas de roble para añadir complejidad."),
                ("4. Tiempo de guarda", "Algunos vinos mejoran significativamente con el añejamiento."),
                ("5. Pruebas de laboratorio", "Realiza análisis químicos regulares para monitorear los parámetros clave.")
            ]
            
            for titulo, descripcion in consejos:
                # Frame para cada consejo
                consejo_frame = ttk.Frame(frame)
                consejo_frame.pack(fill='x', pady=(0, 5), anchor='w')
                
                # Título en negrita
                ttk.Label(
                    consejo_frame,
                    text=f"{titulo}: ",
                    font=('Segoe UI', 10, 'bold')
                ).pack(side='left')
                
                # Descripción normal
                ttk.Label(
                    consejo_frame,
                    text=descripcion,
                    font=('Segoe UI', 10)
                ).pack(side='left')
                
        except Exception as e:
            print(f"Error al generar recomendaciones: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Mostrar mensaje de error
            ttk.Label(
                frame,
                text="No se pudieron cargar las recomendaciones detalladas.",
                foreground='red',
                wraplength=700,
                justify='left',
                padding=(0, 5, 0, 5)
            ).pack(anchor='w')
    
    def add_charts(self):
        """Agrega gráficos al reporte"""
        try:
            if not hasattr(self, 'current_report_data'):
                return
                
            input_data, prediction = self.current_report_data
            
            # Asegurarse de que los directorios necesarios existan
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            
            # Crear un frame para organizar los gráficos en una cuadrícula
            charts_container = ttk.Frame(self.content_frame)
            charts_container.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Fila 1: Gráfico de importancia y comparación
            row1 = ttk.Frame(charts_container)
            row1.pack(fill='x', pady=(0, 20))
            
            # Gráfico de importancia de características (izquierda)
            left_frame = ttk.Frame(row1)
            left_frame.pack(side='left', fill='both', expand=True, padx=5)
            self.add_feature_importance_chart(left_frame)
            
            # Gráfico de comparación (derecha)
            right_frame = ttk.Frame(row1)
            right_frame.pack(side='right', fill='both', expand=True, padx=5)
            self.add_comparison_chart(right_frame, input_data)
            
            # Fila 2: Nuevos gráficos
            row2 = ttk.Frame(charts_container)
            row2.pack(fill='x', pady=(0, 20))
            
            # Gráfico de distribución de calidad (izquierda)
            left_frame2 = ttk.Frame(row2)
            left_frame2.pack(side='left', fill='both', expand=True, padx=5)
            self.add_quality_distribution_chart(left_frame2, input_data, prediction)
            
            # Gráfico de correlación (derecha)
            right_frame2 = ttk.Frame(row2)
            right_frame2.pack(side='right', fill='both', expand=True, padx=5)
            self.add_correlation_chart(right_frame2)
            
        except Exception as e:
            print(f"Error al generar los gráficos: {str(e)}")
            ttk.Label(
                self.content_frame,
                text=f"Error al generar los gráficos: {str(e)}",
                foreground='red',
                wraplength=700
            ).pack(pady=10)
    
    def add_feature_importance_chart(self, parent):
        """Agrega el gráfico de importancia de características mejorado"""
        try:
            # Obtener importancia de características
            importance = self.model.named_steps['model'].feature_importances_
            features = self.X_train.columns
            
            # Ordenar características por importancia
            indices = np.argsort(importance)[::-1]
            features_sorted = [features[i] for i in indices]
            importance_sorted = importance[indices]
            
            # Crear figura con fondo blanco
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            # Crear gráfico de barras horizontales
            bars = ax.barh(
                range(len(indices)),
                importance_sorted,
                color='#722F37',  # Color vino tinto
                alpha=0.8,
                height=0.7
            )
            
            # Añadir etiquetas de valor
            for i, (value, feature) in enumerate(zip(importance_sorted, features_sorted)):
                ax.text(
                    value + 0.005,  # Pequeño desplazamiento a la derecha
                    i,  # Misma altura que la barra
                    f'{value:.2f}',
                    va='center',
                    fontsize=9,
                    color='#333333'
                )
            
            # Personalizar ejes
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(features_sorted, fontsize=10)
            ax.set_xlabel('Importancia Relativa', fontsize=11, labelpad=10)
            ax.set_title('Importancia de las Características', 
                         fontsize=14, fontweight='bold', pad=15)
            
            # Mejorar el aspecto general
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.tick_params(axis='both', which='both', length=0)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Ajustar márgenes
            plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
            
            # Guardar la figura temporalmente
            temp_file = "temp_feature_importance.png"
            plt.tight_layout()
            plt.savefig(temp_file, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Mostrar la imagen en la interfaz
            self.display_image(parent, temp_file, "Importancia de Características")
            
            # Eliminar archivo temporal
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"Error al generar el gráfico de importancia: {str(e)}")
            ttk.Label(
                parent,
                text=f"No se pudo generar el gráfico de importancia: {str(e)}",
                foreground='red'
            ).pack()
    
    def add_comparison_chart(self, parent, input_data):
        """Agrega un gráfico comparativo con valores típicos mejorado"""
        try:
            # Seleccionar las 5 características más importantes
            importance = self.model.named_steps['model'].feature_importances_
            top_indices = np.argsort(importance)[-5:][::-1]
            top_features = [self.X_train.columns[i] for i in top_indices]
            
            # Preparar datos
            means = []
            medians = []
            values = []
            feature_names = []
            
            for feature in top_features:
                if feature not in self.X_train.columns:
                    continue
                    
                stats = self.X_train[feature].describe()
                if '25%' in stats and '75%' in stats and '50%' in stats:
                    means.append((stats['25%'], stats['75%']))  # Rango intercuartílico
                    medians.append(stats['50%'])  # Mediana
                    values.append(input_data.get(feature, np.nan))
                    feature_names.append(feature)
            
            if not feature_names:
                raise ValueError("No se pudieron calcular las estadísticas necesarias")
            
            # Crear figura con estilo
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            
            # Configurar posiciones y dimensiones
            x = np.arange(len(feature_names))
            width = 0.7
            
            # Graficar rango intercuartílico
            lower = [m[0] for m in means]
            upper = [m[1] for m in means]
            
            # Crear gráfico de caja simplificado
            for i in range(len(feature_names)):
                # Dibujar la caja del rango intercuartílico
                ax.plot([i - width/2.5, i + width/2.5], [lower[i], lower[i]], color='#2c3e50', linewidth=1.5)
                ax.plot([i - width/2.5, i + width/2.5], [upper[i], upper[i]], color='#2c3e50', linewidth=1.5)
                ax.plot([i - width/2.5, i - width/2.5], [lower[i], upper[i]], color='#2c3e50', linewidth=1.5)
                ax.plot([i + width/2.5, i + width/2.5], [lower[i], upper[i]], color='#2c3e50', linewidth=1.5)
                
                # Línea vertical para la mediana
                ax.plot([i - width/2.5, i + width/2.5], [medians[i], medians[i]], 
                        color='#e74c3c', linewidth=2, label='Mediana' if i == 0 else '')
            
            # Graficar valores ingresados con colores según si están dentro del rango
            for i, (val, low, high) in enumerate(zip(values, lower, upper)):
                if not np.isnan(val):
                    # Determinar color según si está dentro del rango
                    if low <= val <= high:
                        color = '#2ecc71'  # Verde si está dentro del rango
                        label = 'Dentro del rango típico' if i == 0 else ''
                    else:
                        color = '#e74c3c'  # Rojo si está fuera del rango
                        label = 'Fuera del rango típico' if i == 0 else ''
                    
                    # Dibujar el punto
                    ax.scatter(i, val, color=color, s=120, zorder=5, 
                             label=label, edgecolor='white', linewidth=1.5)
                    
                    # Añadir etiqueta con el valor
                    offset = (high - low) * 0.1  # 10% del rango como offset
                    va = 'bottom' if val > medians[i] else 'top'
                    y_pos = val + offset if val > medians[i] else val - offset
                    
                    ax.text(i, y_pos, f'{val:.2f}', 
                           ha='center', va=va, fontsize=9, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.8, 
                                   edgecolor=color, 
                                   boxstyle='round,pad=0.2'))
            
            # Personalizar el gráfico
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
            ax.set_title('Comparación con Valores Típicos', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Valor', fontsize=11, labelpad=10)
            
            # Mejorar el aspecto general
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DDDDDD')
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Añadir leyenda
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)
            
            # Ajustar márgenes
            plt.tight_layout()
            
            # Guardar temporalmente
            temp_file = "temp_comparison.png"
            plt.savefig(temp_file, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Mostrar en la interfaz
            self.display_image(parent, temp_file, "Comparación con Valores Típicos")
            
            # Eliminar archivo temporal
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"Error al generar el gráfico comparativo: {str(e)}")
            ttk.Label(
                parent,
                text=f"No se pudo generar el gráfico comparativo: {str(e)}",
                foreground='red'
            ).pack()
    
    def add_quality_distribution_chart(self, parent, input_data, prediction):
        """Agrega un gráfico de distribución de la calidad del vino"""
        try:
            # Obtener las calidades del conjunto de entrenamiento
            if not hasattr(self, 'y_train'):
                raise ValueError("No se encontraron datos de entrenamiento")
            
            # Crear figura con estilo
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            # Crear histograma de calidades
            unique_qualities = sorted(self.y_train.unique())
            counts = [sum(self.y_train == q) for q in unique_qualities]
            
            # Crear gráfico de barras con degradado
            bars = ax.bar(unique_qualities, counts, 
                         color='#8B0000',  # Rojo vino oscuro
                         alpha=0.7,
                         width=0.8,
                         edgecolor='white',
                         linewidth=1)
            
            # Añadir etiquetas en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}',
                        ha='center', va='bottom',
                        fontsize=9)
            
            # Resaltar la predicción actual
            try:
                if prediction is not None and len(prediction) > 0:
                    pred_quality = round(float(prediction[0]))
                    
                    # Asegurarse de que la calidad predicha existe en los datos
                    if pred_quality in unique_qualities:
                        pred_idx = unique_qualities.index(pred_quality)
                        bars[pred_idx].set_color('#FFD700')  # Dorado para resaltar
                        bars[pred_idx].set_alpha(0.9)
                        
                        # Añadir anotación
                        ax.annotate('Tu predicción',
                                  xy=(pred_quality, counts[pred_idx]),
                                  xytext=(0, 20),
                                  textcoords='offset points',
                                  ha='center',
                                  va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, edgecolor='#FFD700'),
                                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='#FFD700'))
            except (ValueError, IndexError, TypeError) as pred_error:
                print(f"Advertencia al procesar la predicción: {str(pred_error)}")
            
            # Personalizar el gráfico
            ax.set_xlabel('Calidad del Vino', fontsize=11, labelpad=10)
            ax.set_ylabel('Número de Muestras', fontsize=11, labelpad=10)
            ax.set_title('Distribución de Calidad del Vino', 
                        fontsize=14, fontweight='bold', pad=15)
            
            # Ajustar los límites del eje Y para dar espacio a las etiquetas
            ax.set_ylim(0, max(counts) * 1.15)
            
            # Mejorar el aspecto general
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DDDDDD')
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Ajustar márgenes
            plt.tight_layout()
            
            # Guardar temporalmente
            temp_file = "temp_quality_dist.png"
            plt.savefig(temp_file, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Mostrar en la interfaz
            self.display_image(parent, temp_file, "Distribución de Calidad")
            
            # Eliminar archivo temporal
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"Error al generar el gráfico de distribución de calidad: {str(e)}")
            ttk.Label(
                parent,
                text=f"No se pudo generar el gráfico de distribución: {str(e)}",
                foreground='red'
            ).pack()
    
    def display_image(self, parent, image_path, title):
        """Muestra una imagen en la interfaz"""
        try:
            # Asegurar que el directorio de activos exista
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            
            # Crear frame para la imagen
            img_frame = ttk.Frame(parent)
            img_frame.pack(fill='x', pady=10, expand=True)
            
            # Título del gráfico
            ttk.Label(
                img_frame,
                text=title,
                font=('Segoe UI', 10, 'bold')
            ).pack(anchor='w', pady=(0, 5))
            
            # Cargar la imagen
            try:
                img = Image.open(image_path)
                # Calcular el tamaño manteniendo la relación de aspecto
                width, height = img.size
                ratio = min(700/width, 400/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Mostrar la imagen
                label = ttk.Label(img_frame, image=photo)
                label.image = photo  # Mantener una referencia
                label.pack()
            except Exception as img_error:
                print(f"Error al cargar la imagen {image_path}: {str(img_error)}")
                ttk.Label(
                    img_frame,
                    text=f"No se pudo cargar la imagen: {os.path.basename(image_path)}",
                    foreground='red'
                ).pack()
            
        except Exception as e:
            print(f"Error al mostrar la imagen: {str(e)}")
    
    def get_quality_label(self, score):
        """Devuelve la etiqueta de calidad basada en la puntuación"""
        if score >= 7:
            return 'Excelente', '#27ae60'  # Verde
        elif score >= 5:
            return 'Bueno', '#f39c12'     # Naranja
        else:
            return 'Regular', '#e74c3c'   # Rojo
    
    def _on_canvas_configure(self, event):
        # Ajustar el ancho del frame interno al del canvas
        canvas_width = event.width
        self.canvas.itemconfig('inner_frame', width=canvas_width)
        # Actualizar el scrollregion cuando cambia el tamaño
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _on_mousewheel(self, event):
        # Manejar el desplazamiento con la rueda del ratón
        if event.delta:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def get_frame(self):
        """Devuelve el frame principal"""
        return self.main_frame
        
    def add_correlation_chart(self, parent):
        """Agrega un gráfico de correlación entre características"""
        try:
            # Crear un DataFrame con los datos de entrenamiento
            df = self.X_train.copy()
            
            # Calcular la matriz de correlación
            corr = df.corr()
            
            # Crear figura con estilo
            plt.style.use('seaborn-v0_8-white')
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            
            # Crear máscara para el triángulo superior
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Crear mapa de calor de correlación
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                corr, 
                mask=mask,
                cmap=cmap,
                vmax=1, 
                vmin=-1,
                center=0,
                square=True, 
                linewidths=0.5, 
                cbar_kws={"shrink": 0.8},
                annot=True,
                fmt=".2f",
                annot_kws={"size": 9}
            )
            
            # Mejorar la apariencia
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            plt.title('Correlación entre Características', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Ajustar el diseño
            plt.tight_layout()
            
            # Guardar temporalmente
            temp_file = "temp_correlation.png"
            plt.savefig(temp_file, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Mostrar en la interfaz
            self.display_image(parent, temp_file, "Correlación entre Características")
            
            # Eliminar archivo temporal
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"Error al generar el gráfico de correlación: {str(e)}")
            ttk.Label(
                parent,
                text=f"No se pudo generar el gráfico de correlación: {str(e)}",
                foreground='red'
            ).pack()
