import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import os

# Colores y estilos
THEME = {
    'primary': '#2C3E50',
    'secondary': '#34495E',
    'accent': '#3498DB',
    'success': '#27AE60',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'light': '#ECF0F1',
    'dark': '#2C3E50',
    'white': '#FFFFFF',
    'wine_red': '#722F37',
    'wine_white': '#7A6F59',
    'text_primary': '#2C3E50',
    'text_secondary': '#7F8C8D',
    'border': '#BDC3C7'
}

# Estilos
STYLE = {
    'title': ('Segoe UI', 20, 'bold'),
    'subtitle': ('Segoe UI', 14, 'bold'),
    'section': ('Segoe UI', 12, 'bold'),
    'body': ('Segoe UI', 10),
    'caption': ('Segoe UI', 9),
    'card': {
        'bg': THEME['white'],
        'bd': 1,
        'relief': 'solid',
        'padx': 15,
        'pady': 10,
        'highlightthickness': 0
    },
    'card_title': ('Segoe UI', 12, 'bold'),
    'card_body': ('Segoe UI', 10)
}

class InfoScreen:
    """Pantalla de información que muestra contenido según el tipo de vino"""
    
    def __init__(self, parent):
        """Inicializa la pantalla de información"""
        self.parent = parent
        self.frame = ttk.Frame(parent)
        self.wine_type = 'white'  # Valor por defecto
        
        # Configurar estilo de los widgets
        self.configure_styles()
        
        # Crear la interfaz
        self.create_widgets()
    
    def configure_styles(self):
        """Configura los estilos de los widgets"""
        style = ttk.Style()
        
        # Configurar estilo de las pestañas
        style.configure('Custom.TNotebook', background=THEME['light'])
        style.configure('Custom.TNotebook.Tab', 
                       padding=[15, 5], 
                       font=('Segoe UI', 10, 'bold'))
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', THEME['primary'])],
                 foreground=[('selected', THEME['white'])])
    
    def get_frame(self):
        """Devuelve el frame principal"""
        return self.frame
    
    def set_wine_type(self, wine_type):
        """Actualiza el tipo de vino y refresca la información"""
        self.wine_type = wine_type
        self.update_content()
    
    def create_widgets(self):
        """Crea los widgets de la interfaz"""
        # Frame principal con scroll
        self.canvas = tk.Canvas(self.frame, bg=THEME['light'], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configurar el canvas para que se ajuste al ancho
        def on_configure(event):
            # Configurar el ancho del frame interno al ancho del canvas
            self.canvas.itemconfig('inner_frame', width=event.width)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Configurar el frame desplazable
        self.scrollable_frame.bind("<Configure>", on_configure)
        self.canvas.bind('<Configure>', on_configure)
        
        # Crear ventana en el canvas para el frame desplazable
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", tags=('inner_frame',))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Empaquetar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configurar el grid para que se expanda
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # Frame para el contenido que se expande
        self.content_frame = ttk.Frame(self.scrollable_frame, padding=20)
        self.content_frame.pack(fill='x', expand=True)
        self.content_frame.columnconfigure(0, weight=1)  # Hace que la columna 0 se expanda
        
        # Inicializar el contenido
        self.update_content()
        
        # Configurar el evento de la rueda del ratón
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Enter>", lambda _: self.canvas.focus_set())
        self.canvas.bind("<Leave>", lambda _: self.frame.focus_set())
    
    def _on_mousewheel(self, event):
        """Maneja el desplazamiento con la rueda del ratón"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def create_card(self, parent, title, content, icon=None, color=None):
        """Crea una tarjeta con título y contenido"""
        # Frame principal de la tarjeta
        card = tk.Frame(
            parent,
            bg=THEME['white'],
            bd=1,
            relief='solid',
            padx=15,
            pady=10,
            highlightthickness=0
        )
        
        # Configurar color del borde si se especifica
        if color:
            card.config(highlightbackground=color, highlightthickness=1)
        
        # Título de la tarjeta
        title_frame = tk.Frame(card, bg='white')
        title_frame.pack(fill='x', pady=(0, 10))
        
        # Icono si está presente
        if icon:
            icon_label = ttk.Label(title_frame, text=icon, font=('Segoe UI', 14))
            icon_label.pack(side='left', padx=(0, 10))
        
        # Título
        title_label = ttk.Label(
            title_frame,
            text=title,
            font=STYLE['card_title'],
            foreground=THEME['primary']
        )
        title_label.pack(side='left')
        
        # Contenido
        if isinstance(content, str):
            content_label = ttk.Label(
                card,
                text=content,
                font=STYLE['card_body'],
                wraplength=0,  # Permite que el texto se ajuste al ancho
                justify='left',
                anchor='w'
            )
            content_label.pack(fill='x', anchor='w')
        elif isinstance(content, tk.Widget):
            content.pack(in_=card, fill='both', expand=True)
        
        return card
    
    def create_info_table(self, parent, data, columns=2):
        """Crea una tabla de información con múltiples columnas"""
        frame = tk.Frame(parent, bg='white')
        
        for i, (key, value) in enumerate(data.items()):
            row = i // columns
            col = (i % columns) * 2
            
            # Etiqueta (nombre de la característica)
            label = ttk.Label(
                frame,
                text=f"{key}:",
                font=('Segoe UI', 9, 'bold'),
                background='white'
            )
            label.grid(row=row, column=col, sticky='w', padx=2, pady=2)
            
            # Valor
            value_label = ttk.Label(
                frame,
                text=value,
                font=('Segoe UI', 9),
                background='white'
            )
            value_label.grid(row=row, column=col+1, sticky='w', padx=2, pady=2)
        
        return frame
    
    def update_content(self):
        """Actualiza el contenido según el tipo de vino"""
        # Limpiar contenido anterior
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Determinar el texto según el tipo de vino
        wine_name = "blanco" if self.wine_type == 'white' else "tinto"
        wine_color = THEME['wine_white'] if self.wine_type == 'white' else THEME['wine_red']
        
        # Título principal
        title_frame = tk.Frame(self.content_frame, bg=THEME['light'])
        title_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Label(
            title_frame,
            text=f"Bienvenido al Predictor de Calidad de Vino {wine_name.capitalize()}",
            font=STYLE['title'],
            foreground=wine_color,
            background=THEME['light']
        ).pack(pady=20)
        
        # Sección de descripción
        desc_text = (
            "Esta aplicación utiliza un modelo de aprendizaje automático para predecir "
            f"la calidad del vino {wine_name} basándose en sus características físico-químicas.\n\n"
        )
        
        if self.wine_type == 'white':
            desc_text += (
                "El vino blanco se caracteriza por su acidez más pronunciada y su perfil "
                "aromático frutado y floral. La calidad se evalúa en una escala del 0 al 10."
            )
        else:
            desc_text += (
                "El vino tinto se caracteriza por su cuerpo, taninos y sabores a frutos rojos "
                "y especias. La calidad se evalúa en una escala del 0 al 10."
            )
        
        # Tarjeta de descripción
        desc_card = self.create_card(
            self.content_frame,
            "Descripción",
            desc_text,
            icon="ℹ️"
        )
        desc_card.pack(fill='x', pady=(0, 20), padx=10)  # Añadido padx para margen lateral
        
        # Tarjeta de rangos típicos
        ranges_frame = ttk.Frame(self.content_frame)
        ranges_frame.pack(fill='x', pady=(0, 20), padx=10)  # Añadido padx para margen lateral
        
        # Datos de rangos típicos
        if self.wine_type == 'white':
            ranges_data = {
                'Acidez fija': '3.8 - 14.2 g/L',
                'Acidez volátil': '0.08 - 1.1 g/L',
                'Ácido cítrico': '0 - 1.66 g/L',
                'Azúcar residual': '0.6 - 65.8 g/L',
                'Cloruros': '0.009 - 0.346 g/L',
                'SO₂ libre': '2 - 289 mg/L',
                'SO₂ total': '9 - 440 mg/L',
                'Densidad': '0.987 - 1.039 g/cm³',
                'pH': '2.72 - 3.82',
                'Sulfatos': '0.22 - 1.08 g/L',
                'Alcohol': '8 - 14.2 % vol'
            }
        else:
            ranges_data = {
                'Acidez fija': '4.6 - 15.9 g/L',
                'Acidez volátil': '0.12 - 1.58 g/L',
                'Ácido cítrico': '0 - 1 g/L',
                'Azúcar residual': '0.9 - 15.5 g/L',
                'Cloruros': '0.012 - 0.611 g/L',
                'SO₂ libre': '1 - 72 mg/L',
                'SO₂ total': '6 - 289 mg/L',
                'Densidad': '0.99 - 1.004 g/cm³',
                'pH': '2.74 - 4.01',
                'Sulfatos': '0.33 - 2 g/L',
                'Alcohol': '8.4 - 14.9 % vol'
            }
        
        # Crear tabla de rangos
        ranges_table = self.create_info_table(None, ranges_data, columns=2)
        
        # Crear tarjeta de rangos
        ranges_card = self.create_card(
            ranges_frame,
            "Rangos de Referencia",
            ranges_table,
            icon="📊",
            color=wine_color
        )
        ranges_card.pack(fill='x', padx=10)  # Añadido padx para margen lateral
        
        # Tarjeta de instrucciones
        instructions = (
            "1. Navega a la pestaña 'Predicción'\n"
            "2. Ingresa los valores de las características del vino\n"
            "3. Haz clic en 'Predecir Calidad' para ver el resultado\n"
            "4. Usa 'Aleatorio' para generar valores de ejemplo\n"
            "5. Consulta las estadísticas para más información\n\n"
            "💡 Los valores entre paréntesis indican los rangos aceptados para cada campo.\n"
            "💡 Puedes dejar campos vacíos para valores desconocidos."
        )
        
        instructions_card = self.create_card(
            self.content_frame,
            "Instrucciones de Uso",
            instructions,
            icon="📝"
        )
        instructions_card.pack(fill='x', padx=10)  # Añadido padx para margen lateral
        
        # Consejos adicionales
        tips = (
            "🔍 Consejos para interpretar los resultados:\n\n"
            "• Los vinos con puntuación superior a 7 se consideran de alta calidad.\n"
            "• Valores extremos en acidez o azúcar pueden afectar negativamente la calidad.\n"
            "• El equilibrio entre los componentes es clave para un buen vino.\n"
            "• Los rangos mostrados son referenciales y pueden variar según el estilo."
        )
        
        tips_card = self.create_card(
            self.content_frame,
            "Consejos Prácticos",
            tips,
            icon="💡"
        )
        tips_card.pack(fill='x', pady=(20, 10), padx=10)  # Añadido padx para margen lateral

def create_info_screen(parent, wine_type='white'):
    """Función de conveniencia para crear la pantalla de información"""
    info_screen = InfoScreen(parent)
    info_screen.set_wine_type(wine_type)
    return info_screen.get_frame()
