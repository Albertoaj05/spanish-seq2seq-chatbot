# Carga del dataset de intenciones y aumento de datos.

import json
import random
from pathlib import Path

from .text import normalizar_texto, tokenizar

# Ruta del dataset (data/intents.json).
RUTA_DATOS = Path(__file__).resolve().parents[2] / "data" / "intents.json"


# Cargamos las intenciones del fichero JSON.
def cargar_intenciones(ruta=RUTA_DATOS):
    with open(ruta, encoding="utf-8") as fichero:
        return json.load(fichero)


# Devuelve los pares (frase, respuesta, intención) de un grupo: train, val, test o hard.
def pares(intenciones, grupo):
    resultado = []

    for nombre, datos in intenciones.items():
        respuestas = datos["responses"]

        for i, frase in enumerate(datos[grupo]):
            # En entrenamiento vamos alternando las respuestas válidas para que el modelo las vea todas.
            if grupo == "train":
                respuesta = respuestas[i % len(respuestas)]

            # En el resto basta con una, porque al evaluar se acepta cualquiera de las válidas.
            else:
                respuesta = respuestas[0]

            resultado.append((frase, respuesta, nombre))

    return resultado


# Respuestas que se consideran correctas en cada intención, ya normalizadas.
def respuestas_validas(intenciones):
    validas = {}

    for nombre, datos in intenciones.items():
        validas[nombre] = {normalizar_texto(respuesta) for respuesta in datos["responses"]}

    return validas


# Creamos variaciones sencillas de cada frase: añadir una muletilla al principio,
# una coletilla al final o quitar una palabra. Así el modelo no depende de una frase exacta.
def aumentar(pares_entrenamiento, n_copias=3, semilla=0):
    rng = random.Random(semilla)
    prefijos = ["oye", "una pregunta", "perdona", "chatbot", "hola"]
    sufijos = ["por favor", "chatbot", "porfa"]

    nuevos = []

    for frase, respuesta, nombre in pares_entrenamiento:
        for _ in range(n_copias):
            palabras = tokenizar(frase)

            # Elegimos al azar qué cambio hacemos en esta copia.
            cambio = rng.choice(["prefijo", "sufijo", "quitar"])

            if cambio == "prefijo":
                palabras = [rng.choice(prefijos)] + palabras
            elif cambio == "sufijo":
                palabras = palabras + [rng.choice(sufijos)]
            elif len(palabras) > 2:
                # Solo quitamos una palabra si la frase tiene más de dos.
                palabras.pop(rng.randrange(len(palabras)))

            nuevos.append((" ".join(palabras), respuesta, nombre))

    # Devolvemos las frases originales más las nuevas.
    return pares_entrenamiento + nuevos
