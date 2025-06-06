import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ...utils.data_utils import quality_label
from ...config import feature_names, translations, valid_ranges

class PredictionScreen:
    def __init__(self, parent, model, stats, on_predict_callback=None):
        self.parent = parent
        self.model = model
        self.stats = stats
        self.on_predict_callback = on_predict_callback
        self.report_callback = None
        self.entries = {}
        self.create_widgets()
    
    def set_report_callback(self, callback):
        """Establece la función de callback para generar reportes"""
        self.report_callback = callback
        
    def update_model(self, model, stats):
        """
        Actualiza el modelo y las estadísticas
        
        Args:
            model: El modelo de scikit-learn actualizado
            stats: Estadísticas actualizadas del conjunto de datos
        """
        try:
            # Actualizar el modelo y las estadísticas
            self.model = model
            self.stats = stats
            
            # Limpiar la interfaz
            self.clear_interface()
            
            # Actualizar cualquier widget que dependa de las estadísticas
            if hasattr(self, 'stats_frame'):
                self.update_stats_display()
                
        except Exception as e:
            print(f"Error al actualizar el modelo: {str(e)}")
            raise
    
    def clear_interface(self):
        """Limpia la interfaz de usuario"""
        # Limpiar los campos de entrada
        for entry in self.entries.values():
            if hasattr(entry, 'delete'):
                entry.delete(0, tk.END)
        
        # Limpiar el resultado
        if hasattr(self, 'result_label'):
            self.result_label.config(text="")
        
        # Limpiar los campos faltantes
        if hasattr(self, 'missing_fields_label'):
            self.missing_fields_label.config(text="")
    
    def create_stats_display(self):
        """Crea la visualización de estadísticas"""
        try:
            if not hasattr(self, 'stats') or not hasattr(self, 'stats_frame') or not self.stats:
                return
                
            # Limpiar el frame de estadísticas
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Obtener estadísticas de calidad (target)
            target_mean = self.stats.get('target_mean', 0)
            target_min = self.stats.get('target_min', 0)
            target_max = self.stats.get('target_max', 0)
            count = self.stats.get('count', 0)
            
            # Formatear el texto de estadísticas
            stats_text = (
                f"Muestras: {int(count):,}\n"
                f"Calidad media: {target_mean:.2f}\n"
                f"Calidad mínima: {target_min:.1f}\n"
                f"Calidad máxima: {target_max:.1f}"
            )
            
            # Mostrar las estadísticas
            ttk.Label(
                self.stats_frame,
                text=stats_text,
                justify=tk.LEFT,
                font=('Consolas', 9)
            ).pack(anchor='w', pady=5)
            
        except Exception as e:
            print(f"Error al mostrar estadísticas: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Mostrar un mensaje de error en la interfaz
            error_frame = ttk.Frame(self.stats_frame)
            error_frame.pack(fill='x', pady=5)
            
            ttk.Label(
                error_frame,
                text="No se pudieron cargar las estadísticas",
                foreground='red',
                justify=tk.LEFT,
                font=('Consolas', 9)
            ).pack(anchor='w')
            
            # Mostrar detalles del error solo en desarrollo
            if hasattr(__builtins__, '__debug__') and __debug__:
                ttk.Label(
                    error_frame,
                    text=f"Error: {str(e)}",
                    foreground='darkred',
                    justify=tk.LEFT,
                    font=('Consolas', 8),
                    wraplength=400
                ).pack(anchor='w', pady=(5, 0))
    
    def update_stats_display(self):
        """Actualiza la visualización de estadísticas si existe"""
        if hasattr(self, 'stats_frame'):
            # Actualizar las estadísticas mostradas
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Volver a crear la visualización de estadísticas
            self.create_stats_display()
    
    def create_widgets(self):
        """Crea los widgets de la pantalla de predicción"""
        # Frame principal
        self.prediction_frame = ttk.Frame(self.parent, padding=20)
        
        # Título
        ttk.Label(
            self.prediction_frame,
            text="Ingrese los parámetros del vino",
            font=('Segoe UI', 16, 'bold'),
            foreground='#722F37'
        ).pack(pady=(0, 20))
        
        # Frame para el formulario
        form_frame = ttk.Frame(self.prediction_frame)
        form_frame.pack(fill='both', expand=True, pady=10)
        
        # Crear el frame de estadísticas
        self.stats_frame = ttk.LabelFrame(
            self.prediction_frame,
            text="Estadísticas del Conjunto de Datos",
            padding=10
        )
        self.stats_frame.pack(fill='x', pady=10, padx=5)
        
        # Inicializar la visualización de estadísticas
        self.create_stats_display()
        
        # Crear campos del formulario en dos columnas
        self.left_form = ttk.Frame(form_frame)
        self.left_form.pack(side='left', fill='both', expand=True, padx=10)
        
        self.right_form = ttk.Frame(form_frame)
        self.right_form.pack(side='right', fill='both', expand=True, padx=10)
        
        # Crear campos de entrada
        self.create_input_fields()
        
        # Frame para botones de acción
        self.button_frame = ttk.Frame(self.prediction_frame)
        self.button_frame.pack(pady=10)
        
        # Botones superiores
        top_button_frame = ttk.Frame(self.button_frame)
        top_button_frame.pack(pady=(0, 10))
        
        ttk.Button(
            top_button_frame,
            text="🔍 Predecir Calidad",
            command=self.predict_quality,
            style='Accent.TButton'
        ).pack(side='left', padx=5, ipadx=15, ipady=5)
        
        ttk.Button(
            top_button_frame,
            text="📊 Generar Reporte",
            command=self.generate_report,
            style='Accent.TButton'
        ).pack(side='left', padx=5, ipadx=15, ipady=5)
        
        # Botones inferiores
        bottom_button_frame = ttk.Frame(self.button_frame)
        bottom_button_frame.pack()
        
        ttk.Button(
            bottom_button_frame,
            text="🧹 Limpiar",
            command=self.clear_form
        ).pack(side='left', padx=5, ipadx=15, ipady=5)
        
        ttk.Button(
            bottom_button_frame,
            text="🎲 Aleatorio",
            command=self.generate_random_values
        ).pack(side='left', padx=5, ipadx=15, ipady=5)
        
        # Frame para mensajes informativos
        info_frame = ttk.Frame(self.prediction_frame)
        info_frame.pack(pady=(0, 10))
        
        ttk.Label(
            info_frame,
            text="💡 Puedes dejar campos vacíos para valores desconocidos",
            font=('Segoe UI', 9, 'italic'),
            foreground='#666666'
        ).pack()
        
        # Frame para mostrar resultados
        self.result_frame = ttk.Frame(self.prediction_frame, padding=15)
        self.result_frame.pack(fill='x', pady=(20, 10), ipady=10)
        
        self.result_label = ttk.Label(self.result_frame, text="", font=('Segoe UI', 14, 'bold'))
        self.result_label.pack()
        
        self.category_label = ttk.Label(self.result_frame, text="", font=('Segoe UI', 16, 'bold'))
        self.category_label.pack(pady=(5, 0))
        
        self.message_label = ttk.Label(self.result_frame, text="", wraplength=600, justify='center')
        self.message_label.pack(pady=(10, 0), padx=20)
        
        # Enfocar el primer campo al iniciar
        if feature_names and feature_names[0] in self.entries:
            self.entries[feature_names[0]].focus_set()
    
    def create_input_fields(self):
        """Crea los campos de entrada del formulario con ejemplos descriptivos"""
        from ...gui.widgets.entry_with_placeholder import add_placeholder, ValidatedEntry
        
        # Ejemplos de valores típicos para cada campo
        example_values = {
            'fixed acidity': 'Ej: 7.0 (manzana verde)',
            'volatile acidity': 'Ej: 0.3 (vinagre suave)',
            'citric acid': 'Ej: 0.3 (ligeramente cítrico)',
            'residual sugar': 'Ej: 2.0 (seco)',
            'chlorides': 'Ej: 0.05 (ligeramente salado)',
            'free sulfur dioxide': 'Ej: 15 (conservante)',
            'total sulfur dioxide': 'Ej: 40 (protección total)',
            'density': 'Ej: 0.995 (ligero)',
            'pH': 'Ej: 3.3 (balance ácido)',
            'sulphates': 'Ej: 0.6 (conservante natural)',
            'alcohol': 'Ej: 10.5 (% vol)'
        }
        
        # Dividir los campos en dos columnas
        half = (len(feature_names) + 1) // 2
        left_fields = feature_names[:half]
        right_fields = feature_names[half:]
        
        # Función para crear un campo de entrada con su ejemplo
        def create_field(parent_frame, field):
            # Frame principal para el campo
            field_frame = ttk.Frame(parent_frame)
            field_frame.pack(fill='x', pady=4, padx=5)
            
            # Frame para la etiqueta y el campo de entrada
            input_frame = ttk.Frame(field_frame)
            input_frame.pack(fill='x')
            
            # Etiqueta del campo
            label = ttk.Label(
                input_frame, 
                text=f"{translations[field]}:",
                width=22,
                anchor='e'
            )
            label.pack(side='left', padx=(0, 8))
            
            # Campo de entrada
            entry = ValidatedEntry(input_frame, width=18, font=('Segoe UI', 10))
            entry.pack(side='right')
            
            # Etiqueta de ejemplo
            example_text = example_values.get(field, "")
            if example_text:
                example_label = ttk.Label(
                    field_frame,
                    text=example_text,
                    font=('Segoe UI', 8),
                    foreground='#666666',
                    anchor='w',
                    padding=(0, 2, 0, 0)
                )
                example_label.pack(fill='x', padx=(100, 0))
                entry.example_label = example_label
            
            # Configurar placeholder con rango
            min_val, max_val = valid_ranges[field]
            placeholder = f"{min_val:.1f} - {max_val:.1f}"
            add_placeholder(entry, placeholder, example_text)
            
            return entry
        
        # Crear campos en la columna izquierda
        for field in left_fields:
            self.entries[field] = create_field(self.left_form, field)
            
        # Crear campos en la columna derecha
        for field in right_fields:
            self.entries[field] = create_field(self.right_form, field)
    
    def get_input_values(self):
        """Obtiene los valores de los campos de entrada, rellenando los vacíos con la media"""
        input_data = {}
        missing_fields = []
        
        # Verificar si tenemos acceso a los datos de entrenamiento
        has_train_data = hasattr(self.parent, 'X_train') and self.parent.X_train is not None
        
        for field in feature_names:
            value = self.entries[field].get()
            min_val, max_val = valid_ranges[field]
            
            if not value or value == f"{min_val} - {max_val}":
                # Si el campo está vacío, intentar usar la media del conjunto de entrenamiento
                if has_train_data and field in self.parent.X_train.columns:
                    try:
                        mean_value = self.parent.X_train[field].mean()
                        input_data[field] = mean_value
                        # Actualizar el campo con el valor de la media
                        self.entries[field].delete(0, tk.END)
                        self.entries[field].insert(0, f"{mean_value:.2f}".replace('.', ','))
                        self.entries[field].config(foreground='#2c3e50')  # Color más oscuro para valores calculados
                        missing_fields.append(f"{translations[field]} (usado valor medio: {mean_value:.2f})")
                    except Exception as e:
                        print(f"Error al calcular la media para {field}: {str(e)}")
                        input_data[field] = np.nan
                        missing_fields.append(translations[field])
                else:
                    input_data[field] = np.nan
                    missing_fields.append(translations[field])
            else:
                try:
                    # Reemplazar comas por puntos y convertir a float
                    input_data[field] = float(value.replace(',', '.'))
                    # Si la conversión fue exitosa, asegurarse de que el texto esté en negro
                    self.entries[field].config(foreground='black')
                except ValueError:
                    messagebox.showerror("Error", f"Valor inválido para {translations[field]}")
                    return None, None
        
        return input_data, missing_fields
    
    def predict_quality(self):
        """
        Realiza la predicción de calidad del vino basada en los datos de entrada
        """
        try:
            # Obtener datos de entrada
            input_data, missing_fields = self.get_input_values()
            if input_data is None:
                return
                
            # Verificar que el modelo esté disponible
            if self.model is None:
                raise ValueError("El modelo no está disponible para realizar predicciones")
                
            # Convertir a array para el modelo
            try:
                X = np.array([input_data[field] for field in feature_names], dtype=float).reshape(1, -1)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error al convertir los datos de entrada: {str(e)}")
            
            # Realizar predicción
            try:
                # Obtener el modelo del diccionario
                model_to_predict = self.model.get('model') if isinstance(self.model, dict) else self.model
                
                if model_to_predict is None:
                    raise ValueError("No se pudo obtener el modelo para realizar la predicción")
                
                prediction_result = model_to_predict.predict(X)
                
                # Asegurarse de que la predicción sea un valor único
                if hasattr(prediction_result, '__len__') and len(prediction_result) > 0:
                    prediction = float(prediction_result[0])
                else:
                    prediction = float(prediction_result)
                    
                # Obtener etiqueta y color para la predicción
                category, color = quality_label(prediction)
                
                # Mostrar resultados
                self.show_prediction_result(prediction, category, color, missing_fields)
                
                # Llamar al callback si existe
                if callable(self.on_predict_callback):
                    self.on_predict_callback(prediction, missing_fields, input_data)
                    
                # Guardar los datos de entrada para el reporte
                self.last_prediction = prediction
                self.last_input_data = input_data
                
            except Exception as e:
                raise ValueError(f"Error al realizar la predicción: {str(e)}")
                
        except Exception as e:
            error_msg = f"Error al procesar la predicción: {str(e)}"
            print(f"Advertencia: {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def generate_report(self):
        """Genera un reporte con los datos actuales"""
        try:
            if not hasattr(self, 'last_prediction') or not hasattr(self, 'last_input_data'):
                messagebox.showinfo("Información", "Primero realice una predicción para generar un reporte.")
                return
                
            # Obtener los valores actuales del formulario
            input_data, _ = self.get_input_values()
            if input_data is None:
                return
            
            # Convertir a diccionario si es necesario
            if not isinstance(input_data, dict):
                input_data = dict(zip(feature_names, input_data))
            
            # Asegurarse de que los valores sean numéricos
            for key, value in input_data.items():
                if isinstance(value, str):
                    try:
                        input_data[key] = float(value.replace(',', '.'))
                    except (ValueError, AttributeError):
                        input_data[key] = np.nan
            
            if self.report_callback:
                self.report_callback(input_data, self.last_prediction)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar el reporte: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def show_prediction_result(self, prediction, category, color, missing_fields):
        """Muestra los resultados de la predicción"""
        self.result_frame.config(style='Result.TFrame')
        self.result_label.config(
            text=f"Puntuación: {prediction:.1f}/10",
            foreground=color
        )
        self.category_label.config(
            text=f"Calidad: {category}",
            foreground=color
        )
        
        # Mensaje según la categoría
        if category == 'Excelente':
            message = "¡Excelente vino! Disfruta de esta excelente elección."
        elif category == 'Bueno':
            message = "Vino de buena calidad. Una excelente opción."
        else:
            message = "Vino de calidad regular. Podría mejorar con un poco de envejecimiento."
        
        # Agregar nota sobre campos faltantes
        if missing_fields:
            fields_str = ", ".join(missing_fields)
            message += f"\n\nNota: Los siguientes campos fueron estimados: {fields_str}"
        
        self.message_label.config(text=message)
    
    def clear_form(self):
        """Limpia el formulario"""
        for field in feature_names:
            if field in self.entries:
                entry = self.entries[field]
                try:
                    entry.delete(0, tk.END)
                    min_val, max_val = valid_ranges[field]
                    entry.insert(0, f"{min_val} - {max_val}")
                    entry.config(foreground="gray")
                except tk.TclError:
                    continue
        
        # Limpiar resultados
        self.result_frame.config(style='TFrame')
        self.result_label.config(text="")
        self.category_label.config(text="")
        self.message_label.config(text="")
        
        # Eliminar cualquier frame de botones existente
        for widget in self.result_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.destroy()
        
        # Enfocar el primer campo
        if feature_names and feature_names[0] in self.entries:
            self.entries[feature_names[0]].focus_set()
    
    def generate_random_values(self):
        """
        Genera valores aleatorios para los campos basados en estadísticas de los datos de entrenamiento.
        Intenta generar valores dentro del rango intercuartílico (25-75%) la mayoría de las veces,
        con una pequeña probabilidad de generar valores atípicos.
        """
        try:
            # Obtener estadísticas de los datos de entrenamiento si están disponibles
            has_stats = hasattr(self.parent, 'X_train') and self.parent.X_train is not None
            
            for field in feature_names:
                if field not in self.entries:
                    continue
                    
                entry = self.entries[field]
                min_val, max_val = valid_ranges[field]
                
                # Generar valor aleatorio
                if has_stats and field in self.parent.X_train.columns:
                    try:
                        # Obtener estadísticas para esta característica
                        stats = self.parent.X_train[field].describe()
                        
                        # Verificar si tenemos los percentiles necesarios
                        has_percentiles = '25%' in stats and '75%' in stats and 'mean' in stats
                        
                        if has_percentiles:
                            # 80% de probabilidad de estar en el rango intercuartílico
                            # 15% de probabilidad de estar en los extremos (pero dentro del rango válido)
                            # 5% de probabilidad de ser un valor atípico controlado
                            rand_choice = np.random.random()
                            
                            if rand_choice < 0.8:  # 80% de probabilidad
                                # Usar distribución normal centrada en la media, dentro del rango intercuartílico
                                mean = stats['mean']
                                std = (stats['75%'] - stats['25%']) / 1.349  # Aproximación de la desviación estándar
                                random_value = np.random.normal(mean, std/2)  # Usamos std/2 para mantener los valores más cerca de la media
                                # Asegurarse de que esté dentro del rango intercuartílico
                                random_value = max(stats['25%'], min(random_value, stats['75%']))
                            elif rand_choice < 0.95:  # 15% de probabilidad
                                # Usar distribución uniforme en los extremos (pero dentro del rango válido)
                                if np.random.random() < 0.5:  # 50% de probabilidad para cada extremo
                                    random_value = np.random.uniform(min_val, stats['25%'])
                                else:
                                    random_value = np.random.uniform(stats['75%'], max_val)
                            else:  # 5% de probabilidad
                                # Generar un valor atípico controlado (pero dentro del rango válido)
                                random_value = np.random.uniform(min_val, max_val)
                            
                            # Asegurar que el valor esté dentro de los límites válidos
                            random_value = max(min_val, min(random_value, max_val))
                            random_value = round(random_value, 2)
                        else:
                            # Si no hay percentiles, usar distribución normal con media en el centro del rango
                            mean = (min_val + max_val) / 2
                            std = (max_val - min_val) / 6  # Cubre aproximadamente 99.7% del rango
                            random_value = np.random.normal(mean, std/2)
                            random_value = max(min_val, min(random_value, max_val))
                            random_value = round(random_value, 2)
                            
                    except Exception as e:
                        # En caso de error, usar el método antiguo
                        print(f"Error al generar valor aleatorio para {field}: {str(e)}")
                        random_value = round(np.random.uniform(min_val, max_val), 2)
                else:
                    # Si no hay datos de entrenamiento, usar el método antiguo
                    random_value = round(np.random.uniform(min_val, max_val), 2)
                
                # Actualizar el campo
                try:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(random_value).replace('.', ','))
                    entry.config(foreground='black')
                except tk.TclError:
                    continue
            
            messagebox.showinfo("Valores aleatorios", "Se han generado valores aleatorios para todos los campos.")
            
        except Exception as e:
            print(f"Error en generate_random_values: {str(e)}")
            messagebox.showerror("Error", "Ocurrió un error al generar valores aleatorios. Por favor, intente nuevamente.")
    
    def get_frame(self):
        """Devuelve el frame principal de la pantalla"""
        return self.prediction_frame
