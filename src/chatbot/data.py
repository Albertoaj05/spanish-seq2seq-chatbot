# Carga del dataset de intenciones y aumento de datos.
#
# El dataset está en data/intents.json. Cada intención (saludo, hora, encoder...) tiene:
#   - "responses": las respuestas válidas.
#   - "train", "val", "test" y "hard": las frases de cada grupo.

import json
import random
from pathlib import Path

from .text import normalizar_texto, tokenizar

# Ruta del dataset, calculada desde este fichero para que funcione desde cualquier carpeta.
RUTA_DATOS = Path(__file__).resolve().parents[2] / "data" / "intents.json"


def cargar_intenciones(ruta=RUTA_DATOS):
    with open(ruta, encoding="utf-8") as fichero:
        return json.load(fichero)


def pares(intenciones, grupo):
    # Devuelve una lista de tuplas (frase, respuesta, intención) para un grupo:
    # "train", "val", "test" o "hard".
    #
    # En entrenamiento vamos alternando las respuestas válidas de cada intención entre sus
    # frases, así el modelo ve todas. En el resto de grupos basta con una de referencia,
    # porque al evaluar se acepta cualquiera de las respuestas válidas.
    resultado = []

    for nombre, datos in intenciones.items():
        respuestas = datos["responses"]

        for i, frase in enumerate(datos[grupo]):
            if grupo == "train":
                respuesta = respuestas[i % len(respuestas)]
            else:
                respuesta = respuestas[0]

            resultado.append((frase, respuesta, nombre))

    return resultado


def respuestas_validas(intenciones):
    # Para cada intención, el conjunto de respuestas que se consideran correctas,
    # ya normalizadas para poder compararlas con lo que genera el modelo.
    validas = {}

    for nombre, datos in intenciones.items():
        validas[nombre] = {normalizar_texto(respuesta) for respuesta in datos["responses"]}

    return validas


def aumentar(pares_entrenamiento, n_copias=3, semilla=0):
    # Aumento de datos muy sencillo: de cada frase hacemos n_copias variaciones.
    # A cada copia le hacemos UNO de estos cambios, elegido al azar:
    #   - una muletilla delante:  "hola"            -> "oye hola"
    #   - una coletilla detrás:   "dime la hora"    -> "dime la hora por favor"
    #   - quitar una palabra:     "que es el encoder" -> "que es encoder"
    # Es lo que pasa cuando alguien escribe de verdad: casi nunca usa la frase exacta.
    rng = random.Random(semilla)
    prefijos = ["oye", "una pregunta", "perdona", "chatbot", "hola"]
    sufijos = ["por favor", "chatbot", "porfa"]

    nuevos = []

    for frase, respuesta, nombre in pares_entrenamiento:
        for _ in range(n_copias):
            palabras = tokenizar(frase)
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
