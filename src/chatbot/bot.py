"""Clase Chatbot: junta vocabularios, modelo, entrenamiento y generación de respuestas."""

import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences

from .model import construir_seq2seq, generar
from .text import tokenizar
from .vocab import (END, START, VocabularioPreentrenado, vocabulario_decoder,
                    vocabulario_encoder)


class Chatbot:
    def __init__(self, atencion=False, preentrenado=False, emb_dim=64, unidades=128, dropout=0.2,
                 semilla=42):
        self.atencion = atencion
        self.preentrenado = preentrenado
        self.emb_dim, self.unidades, self.dropout = emb_dim, unidades, dropout
        self.semilla = semilla

    # ------------------------------------------------------------------ datos
    def _codificar_preguntas(self, frases):
        return pad_sequences([self.vocab_enc.codificar(f) for f in frases],
                             maxlen=self.max_encoder, padding="post", truncating="post")

    def _codificar_pares(self, pares):
        X_enc = self._codificar_preguntas([f for f, _, _ in pares])
        entrada_dec, salida_dec = [], []
        for _, respuesta, _ in pares:
            ids = [self.vocab_dec.stoi[t] for t in tokenizar(respuesta)]
            entrada_dec.append([self.vocab_dec.stoi[START]] + ids)
            salida_dec.append(ids + [self.vocab_dec.stoi[END]])
        X_dec = pad_sequences(entrada_dec, maxlen=self.max_decoder, padding="post")
        Y = pad_sequences(salida_dec, maxlen=self.max_decoder, padding="post")
        return X_enc, X_dec, Y

    # ----------------------------------------------------------- entrenamiento
    def entrenar(self, pares_train, pares_val=None, epochs=400, paciencia=25, batch_size=8, verbose=0):
        keras.utils.set_random_seed(self.semilla)

        if self.preentrenado:
            self.vocab_enc = VocabularioPreentrenado(pares_train, semilla=self.semilla)
            matriz = self.vocab_enc.matriz
        else:
            self.vocab_enc = vocabulario_encoder(pares_train)
            matriz = None
        self.vocab_dec = vocabulario_decoder(pares_train)

        # Margen de dos palabras para preguntas algo más largas que las de entrenamiento.
        self.max_encoder = max(len(tokenizar(f)) for f, _, _ in pares_train) + 2
        self.max_decoder = max(len(tokenizar(r)) for _, r, _ in pares_train) + 1

        self.modelo, self.modelo_encoder, self.modelo_paso = construir_seq2seq(
            len(self.vocab_enc), len(self.vocab_dec), self.max_encoder, self.max_decoder,
            emb_dim=self.emb_dim, unidades=self.unidades, dropout=self.dropout,
            atencion=self.atencion, matriz_embeddings=matriz)

        X_enc, X_dec, Y = self._codificar_pares(pares_train)
        parametros = dict(batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose)
        if pares_val:
            V_enc, V_dec, V_y = self._codificar_pares(pares_val)
            parametros["validation_data"] = ([V_enc, V_dec], V_y)
            # Paramos cuando la validación deja de mejorar y nos quedamos con los mejores pesos.
            parametros["callbacks"] = [EarlyStopping(monitor="val_loss", patience=paciencia,
                                                     restore_best_weights=True)]
        self.historial = self.modelo.fit([X_enc, X_dec], Y, **parametros).history
        return self

    # --------------------------------------------------------------- respuestas
    def responder_lote(self, frases, devolver_atencion=False):
        ids = self._codificar_preguntas(frases).astype("int32")
        generados, pesos = generar(self.modelo_encoder, self.modelo_paso, ids,
                                   self.vocab_dec.stoi[START], self.vocab_dec.stoi[END],
                                   self.max_decoder, self.atencion)
        respuestas = []
        for fila in generados:
            palabras = []
            for i in fila:
                token = self.vocab_dec.itos[i]
                if token in (END, "<pad>"):
                    break
                palabras.append(token)
            respuestas.append(" ".join(palabras))
        if devolver_atencion:
            return respuestas, pesos
        return respuestas

    def responder(self, frase):
        return self.responder_lote([frase])[0]

    def n_parametros(self):
        entrenables = sum(int(np.prod(w.shape)) for w in self.modelo.trainable_weights)
        return entrenables
