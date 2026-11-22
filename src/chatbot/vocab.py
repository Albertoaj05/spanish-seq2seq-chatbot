"""Vocabularios del encoder y del decoder.

Hay dos formas de construir el vocabulario del encoder:

- A partir de las frases de entrenamiento, con embeddings que se aprenden desde cero.
  Cualquier palabra que no aparezca en entrenamiento se convierte en <unk>.
- A partir de los vectores preentrenados en español de spaCy (es_core_news_md). El
  vocabulario es entonces el de spaCy (unas 350.000 palabras una vez quitadas las tildes), así que una palabra que el
  chatbot no ha visto nunca, como "mano", sigue teniendo un vector con significado.
"""

import numpy as np

from .text import normalizar_texto, tokenizar

PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"


class Vocabulario:
    def __init__(self, tokens_especiales, palabras):
        self.itos = list(tokens_especiales) + sorted(set(palabras) - set(tokens_especiales))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def codificar(self, frase):
        unk = self.stoi.get(UNK, 0)
        return [self.stoi.get(t, unk) for t in tokenizar(frase)]


def vocabulario_encoder(pares):
    palabras = [t for frase, _, _ in pares for t in tokenizar(frase)]
    return Vocabulario([PAD, UNK], palabras)


def vocabulario_decoder(pares):
    palabras = [t for _, respuesta, _ in pares for t in tokenizar(respuesta)]
    return Vocabulario([PAD, START, END], palabras)


class VocabularioPreentrenado:
    """Vocabulario del encoder basado en los vectores de spaCy.

    Fila 0: <pad>. Fila 1: <unk>. Después, los 20.000 vectores distintos de spaCy
    (normalizados a longitud 1) y, al final, una fila extra para cada palabra de
    entrenamiento que spaCy no conoce (por ejemplo "seq2seq" o "lstm").
    """

    def __init__(self, pares_entrenamiento, modelo_spacy="es_core_news_md", semilla=0):
        import spacy

        nlp = spacy.load(modelo_spacy, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                "attribute_ruler", "morphologizer", "senter"])
        vectores = nlp.vocab.vectors
        tabla = np.asarray(vectores.data, dtype="float32")
        tabla /= np.linalg.norm(tabla, axis=1, keepdims=True) + 1e-8

        # Palabra normalizada (sin tildes, minúsculas) -> fila de spaCy.
        # Las claves vienen ordenadas por frecuencia, así que nos quedamos con la primera.
        self.stoi = {}
        for clave, fila in vectores.key2row.items():
            palabra = normalizar_texto(nlp.vocab.strings[clave])
            if palabra and " " not in palabra and palabra not in self.stoi:
                self.stoi[palabra] = fila + 2

        # Palabras de entrenamiento que spaCy no tiene: les damos un vector aleatorio propio.
        rng = np.random.default_rng(semilla)
        extra = []
        for frase, _, _ in pares_entrenamiento:
            for t in tokenizar(frase):
                if t not in self.stoi:
                    self.stoi[t] = 2 + len(tabla) + len(extra)
                    v = rng.normal(size=tabla.shape[1]).astype("float32")
                    extra.append(v / np.linalg.norm(v))

        filas = [np.zeros((2, tabla.shape[1]), dtype="float32"), tabla]
        if extra:
            filas.append(np.stack(extra))
        self.matriz = np.concatenate(filas)
        self.stoi[PAD], self.stoi[UNK] = 0, 1
        self.palabras_extra = len(extra)

    def __len__(self):
        return len(self.matriz)

    def codificar(self, frase):
        return [self.stoi.get(t, 1) for t in tokenizar(frase)]
