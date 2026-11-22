"""Referencia no generativa: buscar la frase de entrenamiento más parecida.

No genera texto: representa cada frase como la media de los vectores de spaCy de sus
palabras y devuelve la respuesta de la frase de entrenamiento más cercana (similitud
coseno). Sirve para saber qué parte de la dificultad viene de entender la pregunta y
qué parte de generar la respuesta palabra a palabra.
"""

import numpy as np

from .text import tokenizar


class Recuperacion:
    def __init__(self, modelo_spacy="es_core_news_md"):
        import spacy
        self.nlp = spacy.load(modelo_spacy, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                     "attribute_ruler", "morphologizer", "senter"])

    def _vector(self, frase):
        vectores = [self.nlp.vocab[t].vector for t in tokenizar(frase) if self.nlp.vocab[t].has_vector]
        if not vectores:
            return np.zeros(self.nlp.vocab.vectors.shape[1], dtype="float32")
        v = np.mean(vectores, axis=0)
        return v / (np.linalg.norm(v) + 1e-8)

    def entrenar(self, pares_train):
        self.respuestas = [r for _, r, _ in pares_train]
        self.matriz = np.stack([self._vector(f) for f, _, _ in pares_train])
        return self

    def responder_lote(self, frases):
        consultas = np.stack([self._vector(f) for f in frases])
        mas_cercana = np.argmax(consultas @ self.matriz.T, axis=1)
        return [" ".join(tokenizar(self.respuestas[i])) for i in mas_cercana]
