# Vocabularios del encoder y del decoder.
#
# La red no trabaja con palabras, sino con números: cada palabra tiene un índice.
# Hay dos formas de construir el vocabulario del encoder:
#
#   1. Con las palabras de entrenamiento (como en la primera versión). El modelo aprende
#      sus embeddings desde cero y cualquier palabra que no haya visto se convierte en <unk>.
#
#   2. Con los vectores preentrenados en español de spaCy (es_core_news_md). El vocabulario
#      pasa a tener unas 350.000 palabras, así que una palabra que el chatbot no ha visto
#      nunca, como "define", sigue teniendo un vector con significado.

import numpy as np

from .text import normalizar_texto, tokenizar

# Tokens especiales.
PAD = "<pad>"       # relleno para que todas las frases midan lo mismo (siempre índice 0)
UNK = "<unk>"       # palabra desconocida
START = "<start>"   # marca el inicio de la respuesta en el decoder
END = "<end>"       # marca el final de la respuesta


class Vocabulario:
    # Vocabulario sencillo: lista de palabras (índice -> palabra) y diccionario (palabra -> índice).

    def __init__(self, tokens_especiales, palabras):
        # Los tokens especiales van primero, así <pad> queda siempre con el índice 0.
        palabras_ordenadas = sorted(set(palabras) - set(tokens_especiales))
        self.itos = list(tokens_especiales) + palabras_ordenadas

        self.stoi = {}
        for indice, token in enumerate(self.itos):
            self.stoi[token] = indice

    def __len__(self):
        return len(self.itos)

    def codificar(self, frase):
        # Cambia cada palabra por su índice. Si no está en el vocabulario, usamos <unk>.
        indice_unk = self.stoi.get(UNK, 0)
        return [self.stoi.get(token, indice_unk) for token in tokenizar(frase)]


def vocabulario_encoder(pares):
    # Palabras que aparecen en las preguntas.
    palabras = []
    for frase, _, _ in pares:
        palabras.extend(tokenizar(frase))

    return Vocabulario([PAD, UNK], palabras)


def vocabulario_decoder(pares):
    # Palabras que aparecen en las respuestas.
    palabras = []
    for _, respuesta, _ in pares:
        palabras.extend(tokenizar(respuesta))

    return Vocabulario([PAD, START, END], palabras)


class VocabularioPreentrenado:
    # Vocabulario del encoder basado en los vectores de spaCy.
    #
    # La matriz de embeddings queda así:
    #   - fila 0: <pad> (ceros)
    #   - fila 1: <unk> (ceros)
    #   - después: los 20.000 vectores distintos de spaCy, normalizados a longitud 1
    #   - al final: una fila para cada palabra de entrenamiento que spaCy no conoce
    #     (por ejemplo "seq2seq" o "lstm"), con un vector aleatorio propio.

    def __init__(self, pares_entrenamiento, modelo_spacy="es_core_news_md", semilla=0):
        import spacy

        # Solo nos hacen falta los vectores, así que no cargamos el resto del modelo.
        nlp = spacy.load(modelo_spacy, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                "attribute_ruler", "morphologizer", "senter"])
        vectores = nlp.vocab.vectors

        # Normalizamos cada vector a longitud 1: lo que importa es su dirección.
        tabla = np.asarray(vectores.data, dtype="float32")
        tabla = tabla / (np.linalg.norm(tabla, axis=1, keepdims=True) + 1e-8)

        # Diccionario palabra (sin tildes, en minúsculas) -> fila de la matriz.
        # Las palabras de spaCy vienen ordenadas por frecuencia, así que si dos palabras
        # quedan iguales al quitar las tildes, nos quedamos con la más frecuente.
        # Sumamos 2 porque las dos primeras filas son <pad> y <unk>.
        self.stoi = {}
        for clave, fila in vectores.key2row.items():
            palabra = normalizar_texto(nlp.vocab.strings[clave])
            if palabra and " " not in palabra and palabra not in self.stoi:
                self.stoi[palabra] = fila + 2

        # Palabras de entrenamiento que spaCy no conoce: les damos un vector aleatorio.
        rng = np.random.default_rng(semilla)
        vectores_extra = []

        for frase, _, _ in pares_entrenamiento:
            for token in tokenizar(frase):
                if token not in self.stoi:
                    self.stoi[token] = 2 + len(tabla) + len(vectores_extra)
                    vector = rng.normal(size=tabla.shape[1]).astype("float32")
                    vectores_extra.append(vector / np.linalg.norm(vector))

        # Juntamos todo en una sola matriz.
        filas = [np.zeros((2, tabla.shape[1]), dtype="float32"), tabla]
        if vectores_extra:
            filas.append(np.stack(vectores_extra))
        self.matriz = np.concatenate(filas)

        self.stoi[PAD] = 0
        self.stoi[UNK] = 1
        self.palabras_extra = len(vectores_extra)

    def __len__(self):
        return len(self.matriz)

    def codificar(self, frase):
        # Igual que en el vocabulario sencillo: palabra -> índice, y <unk> (1) si no existe.
        return [self.stoi.get(token, 1) for token in tokenizar(frase)]
