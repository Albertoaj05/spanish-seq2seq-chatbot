# Limpieza del texto antes de pasárselo al modelo.
#
# Es el mismo preprocesado que usaba en la primera versión del chatbot: todo en minúsculas,
# sin tildes y sin signos de puntuación. Así "¿Cómo estás?" y "como estas" acaban siendo
# exactamente la misma frase.

import re
import unicodedata


def quitar_tildes(texto):
    # Separamos cada letra de su tilde (á -> a + ´) y nos quedamos solo con las letras.
    texto_separado = unicodedata.normalize("NFD", texto)

    letras = []
    for caracter in texto_separado:
        if unicodedata.category(caracter) != "Mn":   # "Mn" son las tildes sueltas
            letras.append(caracter)

    return "".join(letras)


def normalizar_texto(texto):
    # Pasamos todo a minúsculas y quitamos las tildes.
    texto = texto.lower().strip()
    texto = quitar_tildes(texto)

    # Cambiamos los signos de puntuación por espacios.
    texto = re.sub(r"[¿?¡!,.;:()\"']", " ", texto)

    # Dejamos solo un espacio entre palabras.
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def tokenizar(texto):
    # Convertimos una frase en una lista de palabras.
    # Ejemplo: "¿Cómo estás?" -> ["como", "estas"]
    return normalizar_texto(texto).split()
