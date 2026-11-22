# Clase Chatbot: junta los vocabularios, el modelo, el entrenamiento y la generación
# de respuestas en un solo sitio.
#
# Uso básico:
#     bot = Chatbot(atencion=True, preentrenado=True)
#     bot.entrenar(pares_train, pares_val)
#     bot.responder("¿Qué es el encoder?")

import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences

from .model import construir_seq2seq, generar
from .text import tokenizar
from .vocab import END, PAD, START, VocabularioPreentrenado, vocabulario_decoder, vocabulario_encoder


class Chatbot:

    def __init__(self, atencion=False, preentrenado=False, emb_dim=64, unidades=128, dropout=0.2,
                 semilla=42):
        # atencion: activa la capa de atención en el decoder.
        # preentrenado: usa los vectores de spaCy en el encoder.
        # emb_dim y unidades: tamaño de los embeddings (64) y de las LSTM (128).
        self.atencion = atencion
        self.preentrenado = preentrenado
        self.emb_dim = emb_dim
        self.unidades = unidades
        self.dropout = dropout
        self.semilla = semilla

    # ------------------------------------------------------------ Preparar datos

    def _codificar_preguntas(self, frases):
        # Pasamos cada pregunta a índices y la rellenamos hasta max_encoder.
        secuencias = [self.vocab_enc.codificar(frase) for frase in frases]
        return pad_sequences(secuencias, maxlen=self.max_encoder, padding="post", truncating="post")

    def _codificar_pares(self, pares):
        # Prepara los tres arrays que necesita el modelo para entrenar:
        #   X_enc: la pregunta                         ["que", "es", "lstm"]
        #   X_dec: la respuesta empezando por <start>  ["<start>", "es", "una", ...]
        #   Y:     la respuesta terminando en <end>    ["es", "una", ..., "<end>"]
        X_enc = self._codificar_preguntas([frase for frase, _, _ in pares])

        entrada_decoder = []
        salida_decoder = []

        for _, respuesta, _ in pares:
            indices = [self.vocab_dec.stoi[token] for token in tokenizar(respuesta)]
            entrada_decoder.append([self.vocab_dec.stoi[START]] + indices)
            salida_decoder.append(indices + [self.vocab_dec.stoi[END]])

        X_dec = pad_sequences(entrada_decoder, maxlen=self.max_decoder, padding="post")
        Y = pad_sequences(salida_decoder, maxlen=self.max_decoder, padding="post")

        return X_enc, X_dec, Y

    # -------------------------------------------------------------- Entrenamiento

    def entrenar(self, pares_train, pares_val=None, epochs=400, paciencia=25, batch_size=8, verbose=0):
        # Semilla fija para que el entrenamiento se pueda repetir con los mismos resultados.
        keras.utils.set_random_seed(self.semilla)

        # 1. Vocabularios.
        if self.preentrenado:
            self.vocab_enc = VocabularioPreentrenado(pares_train, semilla=self.semilla)
            matriz = self.vocab_enc.matriz
        else:
            self.vocab_enc = vocabulario_encoder(pares_train)
            matriz = None
        self.vocab_dec = vocabulario_decoder(pares_train)

        # 2. Longitudes máximas. Dejamos un margen de dos palabras en las preguntas por si
        #    el usuario escribe una frase algo más larga que las de entrenamiento.
        self.max_encoder = max(len(tokenizar(frase)) for frase, _, _ in pares_train) + 2
        self.max_decoder = max(len(tokenizar(respuesta)) for _, respuesta, _ in pares_train) + 1

        # 3. Modelo.
        self.modelo, self.modelo_encoder, self.modelo_paso = construir_seq2seq(
            len(self.vocab_enc), len(self.vocab_dec), self.max_encoder, self.max_decoder,
            emb_dim=self.emb_dim, unidades=self.unidades, dropout=self.dropout,
            atencion=self.atencion, matriz_embeddings=matriz,
        )

        # 4. Datos de entrenamiento.
        X_enc, X_dec, Y = self._codificar_pares(pares_train)
        parametros = {"batch_size": batch_size, "epochs": epochs, "shuffle": True, "verbose": verbose}

        # 5. Si hay validación, paramos cuando deja de mejorar y nos quedamos con los mejores pesos.
        if pares_val:
            V_enc, V_dec, V_y = self._codificar_pares(pares_val)
            parametros["validation_data"] = ([V_enc, V_dec], V_y)
            parametros["callbacks"] = [
                EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)
            ]

        # 6. Entrenamos y guardamos las curvas de pérdida y accuracy.
        self.historial = self.modelo.fit([X_enc, X_dec], Y, **parametros).history
        return self

    # ---------------------------------------------------------------- Respuestas

    def responder_lote(self, frases, devolver_atencion=False):
        # Genera la respuesta de una lista de frases a la vez.
        ids = self._codificar_preguntas(frases).astype("int32")
        generados, pesos = generar(
            self.modelo_encoder, self.modelo_paso, ids,
            self.vocab_dec.stoi[START], self.vocab_dec.stoi[END],
            self.max_decoder, self.atencion,
        )

        # Pasamos los índices generados a palabras, parando en <end>.
        respuestas = []
        for fila in generados:
            palabras = []
            for indice in fila:
                token = self.vocab_dec.itos[indice]
                if token in (END, PAD):
                    break
                palabras.append(token)
            respuestas.append(" ".join(palabras))

        if devolver_atencion:
            return respuestas, pesos
        return respuestas

    def responder(self, frase):
        return self.responder_lote([frase])[0]

    def n_parametros(self):
        # Solo cuenta los pesos que se entrenan (los embeddings congelados no cuentan).
        return sum(int(np.prod(peso.shape)) for peso in self.modelo.trainable_weights)
