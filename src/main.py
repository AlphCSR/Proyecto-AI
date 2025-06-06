import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
import joblib
import os
import time
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import webbrowser
from io import BytesIO
import base64

# Importaciones de módulos propios
from src.config import APP_TITLE, PRIMARY_COLOR, SECONDARY_COLOR, WHITE, TEXT_COLOR, feature_names, translations
from src.utils.data_utils import load_training_data, calculate_statistics, load_model
from src.utils.report_utils import generate_prediction_report
from src.gui.screens.prediction_screen import PredictionScreen
from src.gui.screens.report_screen import ReportScreen
from src.gui.screens.info_screen import InfoScreen
from src.gui.screens.training_screen import TrainingScreen

class WineQualityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana principal
        self.title(APP_TITLE)
        self.geometry("1280x1020")
        self.minsize(1280, 1020)
        
        # Crear directorio de activos si no existe
        os.makedirs('assets', exist_ok=True)
        
        # Inicializar variables de instancia
        self.wine_type = 'white'  # Valor por defecto
        self.model = None
        self.X_train = None
        self.y_train = None
        self.stats = None
        self.report_screen = None
        
        # Configurar estilos
        self.setup_styles()
        
        # Crear el frame principal
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Crear el selector de tipo de vino
        self.create_wine_type_selector()
        
        # Configurar menú
        self.setup_menu()
        
        # Crear la barra de estado
        self.status_bar = ttk.Label(
            self.main_frame,
            text="Listo",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        # Crear el notebook
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, pady=(10, 0))
        
        # Cargar recursos iniciales (esto creará las pestañas)
        self.load_resources()
        
        # Generar gráficos iniciales (después de crear la interfaz)
        self.after(100, self.generate_initial_charts)
    
    def create_wine_type_selector(self):
        """Crea el selector de tipo de vino"""
        selector_frame = ttk.Frame(self.main_frame)
        selector_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(selector_frame, text="Tipo de vino:", font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        # Variable para el selector
        self.wine_type_var = tk.StringVar(value=self.wine_type)
        
        # Estilo para los botones de selección
        style = ttk.Style()
        style.configure('WineType.TRadiobutton', padding=5)
        
        # Opción para vino blanco
        ttk.Radiobutton(
            selector_frame, 
            text="Blanco 🍾", 
            variable=self.wine_type_var, 
            value='white',
            command=self.on_wine_type_changed,
            style='WineType.TRadiobutton'
        ).pack(side='left', padx=5)
        
        # Opción para vino tinto
        ttk.Radiobutton(
            selector_frame, 
            text="Tinto 🍷", 
            variable=self.wine_type_var, 
            value='red',
            command=self.on_wine_type_changed,
            style='WineType.TRadiobutton'
        ).pack(side='left', padx=5)
    
    def on_wine_type_changed(self):
        """Se ejecuta cuando se cambia el tipo de vino"""
        new_wine_type = self.wine_type_var.get()
        if new_wine_type != self.wine_type:
            self.wine_type = new_wine_type
            # Recargar recursos con el nuevo tipo de vino
            self.load_resources()
            # No es necesario llamar a update_tabs aquí ya que load_resources() ya lo hace
            # Regenerar gráficos
            self.after(100, self.generate_initial_charts)
    
    def update_tabs(self):
        """Actualiza las pestañas con los nuevos datos"""
        # Guardar referencia a la pestaña actual
        current_tab = self.notebook.select()
        
        # Destruir todas las pestañas
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        
        # Volver a crear las pestañas
        self.create_tabs()
        
        # Actualizar la información del tipo de vino después de recrear las pestañas
        if hasattr(self, 'info_screen'):
            self.info_screen.set_wine_type(self.wine_type)
        
        # Restaurar la pestaña seleccionada si es posible
        try:
            if current_tab and current_tab in self.notebook.tabs():
                self.notebook.select(current_tab)
            else:
                # Si la pestaña ya no existe, seleccionar la primera
                self.notebook.select(0)
        except Exception as e:
            print(f"Error al restaurar la pestaña: {e}")
            self.notebook.select(0)
    
    def load_resources(self):
        """Carga los recursos necesarios (modelo, datos, etc.) para el tipo de vino actual"""
        try:
            from src.config import WINE_TYPES
            
            # Mostrar mensaje de carga
            self.status_bar.config(text=f"Cargando recursos para vino {self.wine_type}...")
            self.update()
            
            # Obtener rutas según el tipo de vino
            wine_config = WINE_TYPES.get(self.wine_type, WINE_TYPES['white'])
            
            # Cargar modelo específico para el tipo de vino
            model_data = load_model(wine_config['model_path'])
            
            # Verificar que el modelo se cargó correctamente
            if model_data is None or 'model' not in model_data:
                raise Exception("No se pudo cargar el modelo correctamente")
                
            self.model = model_data
            
            # Cargar datos de entrenamiento para estadísticas
            self.X_train, self.y_train = load_training_data(wine_config['data_url'])
            self.stats = calculate_statistics(self.X_train, self.y_train)
            
            # Actualizar el título de la ventana según el tipo de vino
            wine_name = "Blanco" if self.wine_type == 'white' else "Tinto"
            self.title(f"{APP_TITLE} - Vino {wine_name}")
            
            # Crear o actualizar pestañas
            if not hasattr(self, 'tabs_created') or not self.tabs_created:
                self.create_tabs()
                self.tabs_created = True
            else:
                self.update_tabs()
            
            # Actualizar estado
            self.status_bar.config(text=f"Recursos cargados para vino {wine_name}")
            
            # Generar gráficos después de cargar todo
            self.after(100, self.generate_initial_charts)
            
        except Exception as e:
            error_msg = f"Error al cargar los recursos para vino {self.wine_type}: {str(e)}"
            print(error_msg)
            self.status_bar.config(text=error_msg)
            
            # Mostrar mensaje de error en la interfaz
            messagebox.showerror(
                "Error de Carga",
                f"No se pudieron cargar los recursos para vino {self.wine_type}.\n"
                f"Error: {str(e)}"
            )
            
            # Intentar cargar el vino por defecto si no es el actual
            if self.wine_type != 'white':
                self.wine_type = 'white'
                self.wine_type_var.set('white')
                self.after(100, self.load_resources)
            else:
                # Si ya estamos en el vino por defecto, mostrar error crítico
                messagebox.showerror(
                    "Error Crítico",
                    "No se pudieron cargar los recursos iniciales. La aplicación se cerrará."
                )
                self.after(1000, self.quit)
    
    def generate_initial_charts(self):
        """
        Función para generar gráficos iniciales (actualmente deshabilitada).
        Se mantiene la estructura por compatibilidad, pero no genera gráficos.
        """
        print("Generación de gráficos deshabilitada")
    
    def setup_menu(self):
        """Configura la barra de menú"""
        menubar = tk.Menu(self)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir reporte...", command=self.open_report)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.quit)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        
        self.config(menu=menubar)
    
    # Función eliminada: show_chart
    
    def open_report(self):
        """Abre un diálogo para seleccionar un reporte"""
        file_path = filedialog.askopenfilename(
            title="Abrir Reporte",
            filetypes=[("Archivos Markdown", "*.md"), ("Todos los archivos", "*.*")],
            initialdir=os.path.abspath('reports')
        )
        
        if file_path:
            try:
                webbrowser.open(f'file://{os.path.abspath(file_path)}')
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir el archivo: {str(e)}")
    
    def setup_styles(self):
        """Configura los estilos de la aplicación"""
        style = ttk.Style()
        
        # Configurar tema
        style.theme_use('clam')
        
        # Configurar colores
        self.configure(bg=SECONDARY_COLOR)
        
        # Configurar estilos de los widgets
        style.configure('TFrame', background=SECONDARY_COLOR)
        style.configure('TLabel', background=SECONDARY_COLOR, foreground=TEXT_COLOR)
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TNotebook', background=SECONDARY_COLOR)
        style.configure('TNotebook.Tab', padding=[10, 5], font=('Segoe UI', 10, 'bold'))
        
        # Estilo para el botón de acción principal
        style.configure('Accent.TButton', 
                       font=('Segoe UI', 12, 'bold'),
                       background=PRIMARY_COLOR, 
                       foreground=WHITE)
        
        style.map('Accent.TButton', 
                 background=[('active', '#5a252b')],
                 foreground=[('active', 'white')])
        
        # Estilo para el marco de resultados
        style.configure('Result.TFrame', background='#f8f9fa', relief='solid', borderwidth=1)
    
    def create_widgets(self):
        """Crea los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Tarjeta principal
        card = ttk.Frame(main_frame, style='Card.TFrame', padding=15)
        card.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título de la aplicación
        ttk.Label(
            card, 
            text="Sistema de predicción de calidad de vino blanco", 
            font=('Segoe UI', 12),
            foreground=TEXT_COLOR,
            background=SECONDARY_COLOR
        ).pack(pady=(5, 0))
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(card)
        self.notebook.pack(fill='both', expand=True)
        
        # Crear pestañas
        self.create_tabs()
    
    def create_tabs(self):
        """Crea las pestañas de la aplicación"""
        try:
            # Limpiar pestañas existentes si las hay
            for tab_id in self.notebook.tabs():
                try:
                    self.notebook.forget(tab_id)
                except:
                    continue
            
            # Verificar que los recursos estén cargados
            if self.model is None or self.stats is None:
                raise ValueError("Los recursos del modelo no están cargados correctamente")
            
            # Pestaña de información
            try:
                self.info_screen = InfoScreen(self.notebook)
                self.notebook.add(self.info_screen.get_frame(), text="   Información   ")
            except Exception as e:
                print(f"Error creando pestaña de información: {str(e)}")
                raise
            
            # Pestaña de predicción
            try:
                self.prediction_screen = PredictionScreen(
                    self.notebook,
                    self.model,
                    self.stats,
                    on_predict_callback=self.on_prediction_made
                )
                self.notebook.add(self.prediction_screen.prediction_frame, text="   Predicción   ")
                
                # Configurar generación de reportes
                self.prediction_screen.set_report_callback(self.generate_prediction_report)
            except Exception as e:
                print(f"Error creando pestaña de predicción: {str(e)}")
                raise
            
            # Pestaña de reportes
            try:
                self.report_tab = ttk.Frame(self.notebook)
                self.notebook.add(self.report_tab, text="   Reporte   ")
            except Exception as e:
                print(f"Error creando pestaña de reportes: {str(e)}")
                raise
            
            # Pestaña de entrenamiento
            try:
                self.training_screen = TrainingScreen(
                    self.notebook,
                    on_training_complete=self.on_training_complete
                )
                self.notebook.add(self.training_screen, text="   Entrenamiento   ")
            except Exception as e:
                print(f"Error creando pestaña de entrenamiento: {str(e)}")
                raise
            
            # Establecer pestaña activa (predicción por defecto)
            try:
                self.notebook.select(1)  # Índice 1 es la pestaña de predicción
            except:
                pass
                
            return True
            
        except Exception as e:
            error_msg = f"Error al crear las pestañas: {str(e)}"
            print(error_msg)
            self.status_bar.config(text=error_msg)
            
            # Mostrar solo la pestaña de error
            for tab_id in self.notebook.tabs():
                self.notebook.forget(tab_id)
                
            error_frame = ttk.Frame(self.notebook)
            ttk.Label(
                error_frame,
                text="Error al cargar la interfaz",
                font=('Segoe UI', 12, 'bold'),
                foreground='red'
            ).pack(pady=20)
            
            ttk.Label(
                error_frame,
                text=str(e),
                wraplength=400
            ).pack(pady=10, padx=20)
            
            ttk.Button(
                error_frame,
                text="Reintentar",
                command=self.load_resources
            ).pack(pady=20)
            
            self.notebook.add(error_frame, text="Error")
            return False
    
    def on_prediction_made(self, prediction, missing_fields, input_data):
        """Callback llamado cuando se realiza una predicción"""
        # Aquí puedes agregar lógica adicional después de una predicción
        pass
    
    def generate_prediction_report(self, input_data, prediction):
        """Genera un reporte de predicción"""
        try:
            # Limpiar el frame de reportes
            for widget in self.report_tab.winfo_children():
                widget.destroy()
            
            # Siempre crear una nueva instancia de ReportScreen para asegurar que use los datos actuales
            self.report_screen = ReportScreen(
                self.report_tab,
                self.model,
                self.X_train,
                self.y_train
            )
            self.report_screen.get_frame().pack(fill='both', expand=True)
            
            # Mostrar el reporte en la pestaña
            self.report_screen.show_report(input_data, prediction)
            
            # Cambiar a la pestaña de reportes
            self.notebook.select(2)  # Índice 2 es la pestaña de reportes
            
            # También guardar el reporte en un archivo
            self.save_report_to_file(input_data, prediction)
            
        except Exception as e:
            error_msg = f"Error al generar el reporte: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def save_report_to_file(self, input_data, prediction):
        """Guarda el reporte en un archivo Markdown"""
        try:
            # Crear directorio de reportes si no existe
            os.makedirs('reports', exist_ok=True)
            
            # Generar reporte
            report_path = generate_prediction_report(
                input_data=input_data,
                prediction=prediction,
                X_train=self.X_train,
                y_train=self.y_train
            )
            
            if report_path and os.path.exists(report_path):
                # Mostrar notificación
                messagebox.showinfo(
                    "Reporte Guardado",
                    f"El reporte se ha guardado en:\n{os.path.abspath(report_path)}"
                )
            
        except Exception as e:
            print(f"Error al guardar el reporte: {str(e)}")
    
    def update_status(self, message):
        """Actualiza la barra de estado si existe"""
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=message)
        self.update_idletasks()
    
    def on_training_complete(self):
        """Maneja la finalización del entrenamiento"""
        try:
            # Mostrar mensaje de carga
            self.update_status("Cargando modelos actualizados...")
            
            # Recargar los recursos
            self.load_resources()
            
            # Actualizar la pestaña de predicción con el nuevo modelo
            if hasattr(self, 'prediction_screen') and self.prediction_screen:
                self.prediction_screen.update_model(self.model, self.stats)
            
            # Actualizar la pestaña de reportes si existe
            if hasattr(self, 'report_screen') and self.report_screen:
                self.report_screen = ReportScreen(
                    self.report_tab,
                    self.model,
                    self.X_train,
                    self.y_train
                )
                self.report_screen.get_frame().pack(fill='both', expand=True)
            
            # Actualizar los gráficos
            self.generate_initial_charts()
            
            # Mostrar mensaje de éxito
            self.update_status("Modelos actualizados correctamente")
            messagebox.showinfo(
                "Entrenamiento Completado",
                "Los modelos se han actualizado correctamente.\n\n"
                "Los cambios ya están disponibles en todas las pestañas."
            )
            
        except Exception as e:
            error_msg = f"Error al cargar los modelos actualizados: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
            # Intentar cargar los modelos originales
            try:
                self.load_resources()
            except Exception as e:
                print(f"Error al cargar modelos originales: {str(e)}")

def main():
    """Función principal de la aplicación"""
    app = WineQualityApp()
    app.mainloop()

if __name__ == "__main__":
    main()
