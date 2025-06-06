import tkinter as tk
from tkinter import ttk

def add_placeholder(entry_widget, placeholder_text, example_text=None):
    """
    Agrega un placeholder y un ejemplo a un widget Entry
    
    Args:
        entry_widget: El widget Entry al que se le agregará el placeholder
        placeholder_text: Texto que se muestra cuando el campo está vacío
        example_text: Texto de ejemplo que se muestra en gris claro (opcional)
    """
    # Configurar el placeholder
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, placeholder_text)
    entry_widget.config(foreground="gray")
    
    # Si se proporciona un texto de ejemplo, mostrarlo en la etiqueta correspondiente
    if hasattr(entry_widget, 'example_label') and example_text:
        entry_widget.example_label.config(text=example_text)

    def on_focus_in(event):
        # Si el texto actual es el placeholder, limpiar el campo
        if entry_widget.get() == placeholder_text:
            entry_widget.delete(0, tk.END)
            entry_widget.config(foreground="black")
            
        # Si hay un ejemplo, resaltarlo
        if hasattr(entry_widget, 'example_label'):
            entry_widget.example_label.config(foreground="#0078D7")  # Azul para indicar foco

    def on_focus_out(event):
        # Si el campo está vacío, restaurar el placeholder
        if entry_widget.get() == "":
            entry_widget.insert(0, placeholder_text)
            entry_widget.config(foreground="gray")
            
        # Restaurar el color del ejemplo
        if hasattr(entry_widget, 'example_label'):
            entry_widget.example_label.config(foreground="#666666")

    # Configurar eventos
    entry_widget.bind("<FocusIn>", on_focus_in, add='+')
    entry_widget.bind("<FocusOut>", on_focus_out, add='+')
    
    return entry_widget

class ValidatedEntry(ttk.Entry):
    """Entry con validación numérica"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._vcmd = (self.register(self._validate), '%P')
        self.config(validate='key', validatecommand=self._vcmd)
        
    def _validate(self, new_value):
        """Valida que la entrada sea numérica"""
        if new_value == "":
            return True
        try:
            # Permitir comas y puntos como separador decimal
            value = new_value.replace(',', '.')
            float(value)
            return True
        except ValueError:
            return False
