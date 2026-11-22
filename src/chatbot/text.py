# Preprocesado del texto: minúsculas, sin tildes y sin signos de puntuación.
# Así "¿Cómo estás?" y "como estas" se convierten en la misma frase.

import re
import unicodedata


# Quitamos tildes para unificar palabras como "Cómo" y "como".
def quitar_tildes(texto):
    texto_normalizado = unicodedata.normalize("NFD", texto)

    letras = []
    for caracter in texto_normalizado:
        if unicodedata.category(caracter) != "Mn":
            letras.append(caracter)

    return "".join(letras)


# Normalizamos el texto antes de dividirlo en palabras.
def normalizar_texto(texto):
    texto = texto.lower().strip()
    texto = quitar_tildes(texto)

    # Cambiamos signos de puntuación por espacios.
    texto = re.sub(r"[¿?¡!,.;:()\"']", " ", texto)

    # Dejamos solo un espacio entre palabras.
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


# Convertimos una frase en una lista de palabras.
def tokenizar(texto):
    return normalizar_texto(texto).split()
