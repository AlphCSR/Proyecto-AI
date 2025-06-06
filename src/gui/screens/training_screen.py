import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os
import sys
import time
from datetime import datetime
from src.config import PRIMARY_COLOR, SECONDARY_COLOR, WHITE, TEXT_COLOR, SECONDARY_LIGHT

class TrainingScreen(ttk.Frame):
    def __init__(self, parent, on_training_complete=None):
        super().__init__(parent)
        self.parent = parent
        self.on_training_complete = on_training_complete
        self.training_in_progress = False
        self.create_widgets()
        
        # Asegurarse de que el directorio de modelos existe
        os.makedirs('src/regressors', exist_ok=True)
    
    def create_widgets(self):
        """Crea la interfaz de usuario para el entrenamiento de modelos"""
        # Frame principal
        self.main_frame = ttk.Frame(self, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(
            self.main_frame,
            text="Entrenamiento de Modelos",
            font=('Helvetica', 16, 'bold'),
            foreground=PRIMARY_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        # Descripción
        desc_text = (
            "Esta herramienta le permite volver a entrenar los modelos de predicción de calidad de vino. "
            "El proceso puede tomar varios minutos dependiendo de su equipo."
        )
        desc_label = ttk.Label(
            self.main_frame,
            text=desc_text,
            wraplength=600,
            justify=tk.LEFT
        )
        desc_label.pack(pady=(0, 30))
        
        # Frame de opciones
        options_frame = ttk.LabelFrame(
            self.main_frame,
            text="Opciones de Entrenamiento",
            padding=15
        )
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Checkbuttons para seleccionar modelos a entrenar
        self.white_var = tk.BooleanVar(value=True)
        self.red_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(
            options_frame,
            text="Modelo de Vino Blanco",
            variable=self.white_var,
            onvalue=True,
            offvalue=False
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Modelo de Vino Tinto",
            variable=self.red_var,
            onvalue=True,
            offvalue=False
        ).pack(anchor=tk.W, pady=5)
        
        # Botón de entrenar
        self.train_button = ttk.Button(
            self.main_frame,
            text="Iniciar Entrenamiento",
            command=self.start_training,
            style="Accent.TButton"
        )
        self.train_button.pack(pady=10)
        
        # Frame para la información de progreso
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Barra de progreso
        self.progress = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=400,
            mode='determinate'
        )
        self.progress.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
        
        # Contador de tiempo
        self.time_label = ttk.Label(progress_frame, text="00:00:00", width=10)
        self.time_label.pack(side=tk.RIGHT)
        self.start_time = None
        
        # Frame para el registro de eventos
        log_frame = ttk.LabelFrame(
            self.main_frame,
            text="Registro de Entrenamiento",
            padding=10
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Área de registro con scroll
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=80,
            height=15,
            font=('Consolas', 9),
            bg='white',
            relief='solid',
            borderwidth=1
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Frame para métricas
        self.metrics_frame = ttk.LabelFrame(
            self.main_frame,
            text="Métricas del Modelo",
            padding=10
        )
        self.metrics_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Variables para las métricas
        self.metrics_vars = {
            'model': tk.StringVar(value="Modelo: -"),
            'status': tk.StringVar(value="Estado: Esperando inicio"),
            'epoch': tk.StringVar(value="Época: -/-"),
            'train_score': tk.StringVar(value="R² (train): -"),
            'test_score': tk.StringVar(value="R² (test): -"),
            'rmse': tk.StringVar(value="RMSE: -"),
            'r2': tk.StringVar(value="R²: -")
        }
        
        # Mostrar las métricas
        for metric, var in self.metrics_vars.items():
            ttk.Label(
                self.metrics_frame,
                textvariable=var,
                anchor='w',
                font=('Segoe UI', 9)
            ).pack(fill=tk.X, pady=2)
        
        # Estilos para el registro
        self.log_text.tag_config('INFO', foreground='black')
        self.log_text.tag_config('SUCCESS', foreground='green')
        self.log_text.tag_config('WARNING', foreground='orange')
        self.log_text.tag_config('ERROR', foreground='red')
        self.log_text.tag_config('HIGHLIGHT', background='#f0f0f0')
    
    def log_message(self, message, level='INFO'):
        """Agrega un mensaje al registro"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()
    
    def update_metrics(self, **kwargs):
        """Actualiza las métricas mostradas"""
        for key, value in kwargs.items():
            if key in self.metrics_vars:
                self.metrics_vars[key].set(f"{key.replace('_', ' ').title()}: {value}")
        self.update_idletasks()
    
    def update_timer(self):
        """Actualiza el contador de tiempo"""
        if self.training_in_progress and self.start_time:
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.time_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            self.after(1000, self.update_timer)
    
    def start_training(self):
        """Inicia el proceso de entrenamiento en un hilo separado"""
        if self.training_in_progress:
            return
            
        if not (self.white_var.get() or self.red_var.get()):
            messagebox.showwarning(
                "Sin modelos seleccionados",
                "Por favor seleccione al menos un modelo para entrenar."
            )
            return
        
        # Reiniciar la interfaz
        self.training_in_progress = True
        self.start_time = time.time()
        self.progress['value'] = 0
        self.time_label.config(text="00:00:00")
        
        # Limpiar el área de registro
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Mostrar estado inicial
        self.log_message("Iniciando proceso de entrenamiento...", 'INFO')
        self.update_metrics(status="Iniciando entrenamiento...")
        
        # Iniciar el contador de tiempo
        self.update_timer()
        
        # Actualizar estado del botón
        self.train_button.config(state=tk.DISABLED)
        
        # Iniciar el entrenamiento en un hilo separado
        training_thread = threading.Thread(
            target=self._run_training_thread,
            daemon=True
        )
        training_thread.start()
        
        # Verificar el estado periódicamente
        self.check_training_status()
    
    def run_training(self):
        """
        Función obsoleta mantenida por compatibilidad.
        El entrenamiento real ahora se maneja directamente en start_training
        """
        self.log_message("Advertencia: run_training() está obsoleto. Usar start_training()", "WARNING")
    
    def _run_training_thread(self):
        """Hilo para ejecutar el entrenamiento"""
        try:
            # Asegurarse de que el directorio de trabajo sea correcto
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
            if os.path.basename(project_root) != 'Proyecto AI':
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                
            # Agregar el directorio raíz al path para poder importar train_models
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            # Cambiar al directorio del proyecto
            original_dir = os.getcwd()
            os.chdir(project_root)
            
            try:
                from train_models import train_wine_model
                
                models_to_train = []
                if self.white_var.get():
                    models_to_train.append(('white', 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'))
                if self.red_var.get():
                    models_to_train.append(('red', 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'))
                
                results = []
                total_models = len(models_to_train)
                
                for i, (model_name, url) in enumerate(models_to_train, 1):
                    current_model = f"{model_name.capitalize()}"
                    self.log_message(f"Iniciando entrenamiento del modelo de vino {current_model}...", 'INFO')
                    self.update_metrics(
                        model=f"Modelo: {current_model}",
                        status=f"Estado: Cargando datos..."
                    )
                    
                    try:
                        # Función de callback para el progreso
                        def progress_callback(progress, message=None, metrics=None):
                            if message:
                                self.log_message(f"{current_model}: {message}", 'INFO')
                            if metrics:
                                self.update_metrics(**metrics)
                            # Actualizar progreso general
                            base_progress = ((i - 1) / total_models) * 100
                            current_model_progress = (progress / 100) * (100 / total_models)
                            total_progress = base_progress + current_model_progress
                            self.after(0, lambda p=total_progress: self.progress.config(value=p))
                            self.update()
                        
                        # Entrenar el modelo con el callback de progreso
                        self.log_message(f"Descargando datos para {current_model}...", 'INFO')
                        result = train_wine_model(
                            url, 
                            model_name,
                            progress_callback=progress_callback
                        )
                        
                        # Registrar resultados
                        results.append((model_name, result))
                        self.log_message(
                            f"Modelo {current_model} entrenado exitosamente. "
                            f"R²: {result['r2']:.4f}, RMSE: {result['rmse']:.4f}", 
                            'SUCCESS'
                        )
                        
                    except Exception as e:
                        import traceback
                        error_msg = f"Error entrenando modelo {current_model}: {str(e)}\n\n{traceback.format_exc()}"
                        self.log_message(error_msg, 'ERROR')
                        self.training_completed(False, error_msg)
                        return
                
                # Mostrar resumen de resultados
                summary = "¡Entrenamiento completado!\n\nResultados:\n"
                for model_name, result in results:
                    summary += f"\nModelo {model_name.capitalize()}:"
                    summary += f"\n  - R² (train): {result['train_score']:.4f}"
                    summary += f"\n  - R² (test): {result['test_score']:.4f}"
                    summary += f"\n  - RMSE: {result['rmse']:.4f}"
                    summary += f"\n  - R²: {result['r2']:.4f}"
                    summary += f"\n  - Guardado en: {result['model_path']}\n"
                
                self.training_completed(True, summary)
                
            except Exception as e:
                import traceback
                error_msg = f"Error durante el entrenamiento: {str(e)}\n\n{traceback.format_exc()}"
                self.log_message(error_msg, 'ERROR')
                self.training_completed(False, error_msg)
                
            finally:
                # Restaurar el directorio original
                os.chdir(original_dir)
                
        except Exception as e:
            import traceback
            error_msg = f"Error inesperado: {str(e)}\n\n{traceback.format_exc()}"
            self.log_message(error_msg, 'ERROR')
            self.training_completed(False, error_msg)
    
    def check_training_status(self):
        """Verifica periódicamente el estado del entrenamiento"""
        if self.training_in_progress:
            self.after(100, self.check_training_status)
    
    def training_completed(self, success, message):
        """
        Maneja la finalización del entrenamiento
        
        Args:
            success: Booleano que indica si el entrenamiento fue exitoso
            message: Mensaje con los resultados o el error
        """
        try:
            # Actualizar el estado de entrenamiento
            self.training_in_progress = False
            
            # Actualizar la barra de progreso
            if hasattr(self, 'progress'):
                self.progress['value'] = 100 if success else 0
            
            # Actualizar estado
            status = "Completado con éxito" if success else "Error en el entrenamiento"
            self.update_metrics(status=f"Estado: {status}")
            
            # Mostrar mensaje en el log
            log_level = 'SUCCESS' if success else 'ERROR'
            self.log_message(f"Entrenamiento {status.lower()}", log_level)
            
            # Mostrar resultados en el área de registro
            self.log_message("\n" + "="*50, 'INFO')
            self.log_message("RESUMEN DE RESULTADOS", 'HIGHLIGHT')
            self.log_message("="*50, 'INFO')
            self.log_message(message, 'INFO')
            
            # Configurar el botón de entrenar
            if hasattr(self, 'train_button'):
                self.train_button.config(state=tk.NORMAL)
            
            # Mostrar mensaje al usuario
            if success:
                # Mostrar mensaje de espera para visualización
                self.log_message("\nEsperando 5 segundos para visualización...", 'INFO')
                self.log_text.see(tk.END)
                self.update()
                
                # Llamar al callback si el entrenamiento fue exitoso
                if hasattr(self, 'on_training_complete') and self.on_training_complete:
                    try:
                        # Esperar 5 segundos antes de llamar al callback
                        self.after(5000, self.on_training_complete)
                    except Exception as e:
                        error_msg = f"Error en el callback de finalización: {str(e)}"
                        self.log_message(error_msg, 'ERROR')
            
            # Hacer scroll hasta el final
            self.log_text.see(tk.END)
            
        except Exception as e:
            # Manejar cualquier error en el propio método de finalización
            error_msg = f"Error al finalizar el entrenamiento: {str(e)}"
            self.log_message(error_msg, 'ERROR')
            print(error_msg)
