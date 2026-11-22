# Referencia sin generar texto: buscar la frase de entrenamiento más parecida.
#
# Cada frase se representa como la media de los vectores de spaCy de sus palabras.
# Para responder, buscamos la frase de entrenamiento más parecida (similitud coseno) y
# devolvemos su respuesta tal cual.
#
# Sirve para saber si generar la respuesta palabra a palabra aporta algo frente a algo
# tan simple como buscar la pregunta más parecida.

import numpy as np

from .text import tokenizar


class Recuperacion:

    def __init__(self, modelo_spacy="es_core_news_md"):
        import spacy
        self.nlp = spacy.load(modelo_spacy, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                     "attribute_ruler", "morphologizer", "senter"])

    def _vector(self, frase):
        # Media de los vectores de las palabras que spaCy conoce, normalizada a longitud 1.
        vectores = []
        for token in tokenizar(frase):
            if self.nlp.vocab[token].has_vector:
                vectores.append(self.nlp.vocab[token].vector)

        if not vectores:
            return np.zeros(self.nlp.vocab.vectors.shape[1], dtype="float32")

        media = np.mean(vectores, axis=0)
        return media / (np.linalg.norm(media) + 1e-8)

    def entrenar(self, pares_train):
        # "Entrenar" aquí es solo guardar el vector de cada frase y su respuesta.
        self.respuestas = [respuesta for _, respuesta, _ in pares_train]
        self.matriz = np.stack([self._vector(frase) for frase, _, _ in pares_train])
        return self

    def responder_lote(self, frases):
        consultas = np.stack([self._vector(frase) for frase in frases])

        # Como todos los vectores miden 1, el producto escalar es la similitud coseno.
        similitudes = consultas @ self.matriz.T
        mas_parecida = np.argmax(similitudes, axis=1)

        return [" ".join(tokenizar(self.respuestas[i])) for i in mas_parecida]
