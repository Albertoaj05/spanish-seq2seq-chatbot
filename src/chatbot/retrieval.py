# Referencia sin generar texto: buscar la frase de entrenamiento más parecida.
# Cada frase se representa con la media de los vectores de spaCy de sus palabras,
# y se devuelve la respuesta de la frase de entrenamiento más cercana.

import numpy as np

from .text import tokenizar


class Recuperacion:

    def __init__(self, modelo_spacy="es_core_news_md"):
        import spacy
        self.nlp = spacy.load(modelo_spacy, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                     "attribute_ruler", "morphologizer", "senter"])

    # Vector de una frase: media de los vectores de sus palabras, normalizada a longitud 1.
    def _vector(self, frase):
        vectores = []
        for token in tokenizar(frase):
            if self.nlp.vocab[token].has_vector:
                vectores.append(self.nlp.vocab[token].vector)

        # Si no conoce ninguna palabra, devolvemos un vector de ceros.
        if not vectores:
            return np.zeros(self.nlp.vocab.vectors.shape[1], dtype="float32")

        media = np.mean(vectores, axis=0)
        return media / (np.linalg.norm(media) + 1e-8)

    # Guardamos el vector de cada frase de entrenamiento y su respuesta.
    def entrenar(self, pares_train):
        self.respuestas = [respuesta for _, respuesta, _ in pares_train]
        self.matriz = np.stack([self._vector(frase) for frase, _, _ in pares_train])
        return self

    # Para cada frase devolvemos la respuesta de la frase de entrenamiento más parecida.
    def responder_lote(self, frases):
        consultas = np.stack([self._vector(frase) for frase in frases])

        # Como todos los vectores miden 1, el producto escalar es la similitud coseno.
        similitudes = consultas @ self.matriz.T
        mas_parecida = np.argmax(similitudes, axis=1)

        return [" ".join(tokenizar(self.respuestas[i])) for i in mas_parecida]
