"""Limpieza y tokenización del texto.

Es el mismo preprocesado que usaba en la primera versión del chatbot: todo a minúsculas,
sin tildes y sin signos de puntuación, para que "¿Cómo estás?" y "como estas" sean la
misma frase.
"""

import re
import unicodedata


def quitar_tildes(texto):
    texto_normalizado = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto_normalizado if unicodedata.category(c) != "Mn")


def normalizar_texto(texto):
    texto = quitar_tildes(texto.lower().strip())
    # Cambiamos signos de puntuación por espacios y dejamos un solo espacio entre palabras.
    texto = re.sub(r"[¿?¡!,.;:()\"']", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def tokenizar(texto):
    return normalizar_texto(texto).split()
