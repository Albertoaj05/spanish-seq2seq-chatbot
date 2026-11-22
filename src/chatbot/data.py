"""Carga del dataset de intenciones y aumento de datos."""

import json
import random
from pathlib import Path

from .text import normalizar_texto, tokenizar

RUTA_DATOS = Path(__file__).resolve().parents[2] / "data" / "intents.json"


def cargar_intenciones(ruta=RUTA_DATOS):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def pares(intenciones, split):
    """Devuelve una lista de tuplas (frase, respuesta, intención) para un split.

    En entrenamiento las respuestas válidas de cada intención se van alternando entre sus
    frases, así el modelo ve todas. En val/test/hard se guarda la primera como referencia;
    al evaluar se acepta cualquiera de las respuestas válidas.
    """
    resultado = []
    for nombre, datos in intenciones.items():
        respuestas = datos["responses"]
        for i, frase in enumerate(datos[split]):
            respuesta = respuestas[i % len(respuestas)] if split == "train" else respuestas[0]
            resultado.append((frase, respuesta, nombre))
    return resultado


def respuestas_validas(intenciones):
    """Intención -> conjunto de respuestas aceptadas (ya normalizadas)."""
    return {
        nombre: {normalizar_texto(r) for r in datos["responses"]}
        for nombre, datos in intenciones.items()
    }


def aumentar(pares_entrenamiento, n_copias=3, semilla=0):
    """Aumento de datos sencillo: a cada frase se le añade una muletilla delante,
    una coletilla detrás o se le quita una palabra. Así el modelo no depende de la
    frase exacta, que es lo que pasa cuando alguien escribe de verdad."""
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
                palabras.pop(rng.randrange(len(palabras)))
            nuevos.append((" ".join(palabras), respuesta, nombre))
    return pares_entrenamiento + nuevos
