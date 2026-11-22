# Vocabularios del encoder y del decoder.
# La red no trabaja con palabras, sino con números, así que cada palabra tiene un índice.

import numpy as np

from .text import normalizar_texto, tokenizar

# Tokens especiales.
PAD = "<pad>"       # relleno para que todas las frases tengan la misma longitud
UNK = "<unk>"       # palabra que no está en el vocabulario
START = "<start>"   # inicio de la respuesta
END = "<end>"       # final de la respuesta


# Vocabulario sencillo: lista de palabras (índice -> palabra) y diccionario (palabra -> índice).
class Vocabulario:

    def __init__(self, tokens_especiales, palabras):
        # Los tokens especiales van al principio, así <pad> tiene siempre el índice 0.
        palabras_ordenadas = sorted(set(palabras) - set(tokens_especiales))
        self.itos = list(tokens_especiales) + palabras_ordenadas

        # Diccionario palabra -> índice.
        self.stoi = {}
        for indice, token in enumerate(self.itos):
            self.stoi[token] = indice

    def __len__(self):
        return len(self.itos)

    # Cambia cada palabra por su índice. Si no está en el vocabulario, se usa <unk>.
    def codificar(self, frase):
        indice_unk = self.stoi.get(UNK, 0)
        return [self.stoi.get(token, indice_unk) for token in tokenizar(frase)]


# Vocabulario del encoder: palabras que aparecen en las preguntas.
def vocabulario_encoder(pares):
    palabras = []
    for frase, _, _ in pares:
        palabras.extend(tokenizar(frase))

    return Vocabulario([PAD, UNK], palabras)


# Vocabulario del decoder: palabras que aparecen en las respuestas.
def vocabulario_decoder(pares):
    palabras = []
    for _, respuesta, _ in pares:
        palabras.extend(tokenizar(respuesta))

    return Vocabulario([PAD, START, END], palabras)


# Vocabulario del encoder con los vectores preentrenados en español de spaCy.
# Tiene unas 350.000 palabras, así que una palabra que el chatbot no ha visto nunca
# sigue teniendo un vector con significado.
class VocabularioPreentrenado:

    def __init__(self, pares_entrenamiento, modelo_spacy="es_core_news_md", semilla=0):
        import spacy

        # Solo necesitamos los vectores, así que no cargamos el resto del modelo de spaCy.
        nlp = spacy.load(modelo_spacy, exclude=["tagger", "parser", "ner", "lemmatizer",
                                                "attribute_ruler", "morphologizer", "senter"])
        vectores = nlp.vocab.vectors

        # Normalizamos cada vector a longitud 1, porque lo que importa es su dirección.
        tabla = np.asarray(vectores.data, dtype="float32")
        tabla = tabla / (np.linalg.norm(tabla, axis=1, keepdims=True) + 1e-8)

        # Diccionario palabra -> fila de la matriz. Sumamos 2 porque las dos primeras filas
        # son <pad> y <unk>. Las palabras vienen ordenadas por frecuencia, así que si dos
        # quedan iguales al quitar las tildes, nos quedamos con la más frecuente.
        self.stoi = {}
        for clave, fila in vectores.key2row.items():
            palabra = normalizar_texto(nlp.vocab.strings[clave])
            if palabra and " " not in palabra and palabra not in self.stoi:
                self.stoi[palabra] = fila + 2

        # Las palabras de entrenamiento que spaCy no conoce (como "seq2seq" o "lstm")
        # reciben un vector aleatorio propio.
        rng = np.random.default_rng(semilla)
        vectores_extra = []

        for frase, _, _ in pares_entrenamiento:
            for token in tokenizar(frase):
                if token not in self.stoi:
                    self.stoi[token] = 2 + len(tabla) + len(vectores_extra)
                    vector = rng.normal(size=tabla.shape[1]).astype("float32")
                    vectores_extra.append(vector / np.linalg.norm(vector))

        # Juntamos todo en una matriz: <pad>, <unk>, vectores de spaCy y vectores extra.
        filas = [np.zeros((2, tabla.shape[1]), dtype="float32"), tabla]
        if vectores_extra:
            filas.append(np.stack(vectores_extra))
        self.matriz = np.concatenate(filas)

        self.stoi[PAD] = 0
        self.stoi[UNK] = 1
        self.palabras_extra = len(vectores_extra)

    def __len__(self):
        return len(self.matriz)

    # Cambia cada palabra por su índice. Si no está, se usa <unk> (índice 1).
    def codificar(self, frase):
        return [self.stoi.get(token, 1) for token in tokenizar(frase)]
