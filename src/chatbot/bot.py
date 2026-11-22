# Clase Chatbot: junta los vocabularios, el modelo, el entrenamiento y las respuestas.
#
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
        # Mejoras que se activan.
        self.atencion = atencion
        self.preentrenado = preentrenado

        # Cada palabra se representa con un vector de 64 valores y las LSTM tienen 128 unidades.
        self.emb_dim = emb_dim
        self.unidades = unidades
        self.dropout = dropout
        self.semilla = semilla

    # Pasamos las preguntas a índices y las rellenamos hasta la longitud máxima.
    def _codificar_preguntas(self, frases):
        secuencias = [self.vocab_enc.codificar(frase) for frase in frases]
        return pad_sequences(secuencias, maxlen=self.max_encoder, padding="post", truncating="post")

    # Preparamos las tres partes que necesita el modelo para entrenar.
    def _codificar_pares(self, pares):
        # Entrada del encoder: la pregunta.
        X_enc = self._codificar_preguntas([frase for frase, _, _ in pares])

        entrada_decoder = []
        salida_decoder = []

        for _, respuesta, _ in pares:
            indices = [self.vocab_dec.stoi[token] for token in tokenizar(respuesta)]

            # La entrada del decoder empieza con <start>.
            entrada_decoder.append([self.vocab_dec.stoi[START]] + indices)

            # La salida esperada termina con <end>.
            salida_decoder.append(indices + [self.vocab_dec.stoi[END]])

        X_dec = pad_sequences(entrada_decoder, maxlen=self.max_decoder, padding="post")
        Y = pad_sequences(salida_decoder, maxlen=self.max_decoder, padding="post")

        return X_enc, X_dec, Y

    def entrenar(self, pares_train, pares_val=None, epochs=400, paciencia=25, batch_size=8, verbose=0):
        # Fijamos la semilla para que los resultados se puedan repetir.
        keras.utils.set_random_seed(self.semilla)

        # Creamos los vocabularios.
        if self.preentrenado:
            self.vocab_enc = VocabularioPreentrenado(pares_train, semilla=self.semilla)
            matriz = self.vocab_enc.matriz
        else:
            self.vocab_enc = vocabulario_encoder(pares_train)
            matriz = None
        self.vocab_dec = vocabulario_decoder(pares_train)

        # Calculamos la longitud máxima de cada parte.
        # En las preguntas dejamos dos palabras de margen por si el usuario escribe algo más largo.
        self.max_encoder = max(len(tokenizar(frase)) for frase, _, _ in pares_train) + 2
        self.max_decoder = max(len(tokenizar(respuesta)) for _, respuesta, _ in pares_train) + 1

        # Construimos el modelo.
        self.modelo, self.modelo_encoder, self.modelo_paso = construir_seq2seq(
            len(self.vocab_enc), len(self.vocab_dec), self.max_encoder, self.max_decoder,
            emb_dim=self.emb_dim, unidades=self.unidades, dropout=self.dropout,
            atencion=self.atencion, matriz_embeddings=matriz,
        )

        # Preparamos los datos de entrenamiento.
        X_enc, X_dec, Y = self._codificar_pares(pares_train)
        parametros = {"batch_size": batch_size, "epochs": epochs, "shuffle": True, "verbose": verbose}

        # Si hay validación, paramos cuando deja de mejorar y nos quedamos con los mejores pesos.
        if pares_val:
            V_enc, V_dec, V_y = self._codificar_pares(pares_val)
            parametros["validation_data"] = ([V_enc, V_dec], V_y)
            parametros["callbacks"] = [
                EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)
            ]

        # Entrenamos y guardamos la evolución de la pérdida y la precisión.
        self.historial = self.modelo.fit([X_enc, X_dec], Y, **parametros).history
        return self

    # Genera la respuesta de varias frases a la vez.
    def responder_lote(self, frases, devolver_atencion=False):
        ids = self._codificar_preguntas(frases).astype("int32")
        generados, pesos = generar(
            self.modelo_encoder, self.modelo_paso, ids,
            self.vocab_dec.stoi[START], self.vocab_dec.stoi[END],
            self.max_decoder, self.atencion,
        )

        # Convertimos los índices generados en palabras.
        respuestas = []
        for fila in generados:
            palabras = []
            for indice in fila:
                token = self.vocab_dec.itos[indice]

                # Al llegar a <end> (o al relleno) la respuesta ha terminado.
                if token in (END, PAD):
                    break

                palabras.append(token)
            respuestas.append(" ".join(palabras))

        if devolver_atencion:
            return respuestas, pesos
        return respuestas

    # Genera la respuesta de una sola frase.
    def responder(self, frase):
        return self.responder_lote([frase])[0]

    # Número de parámetros que se entrenan (los embeddings congelados no cuentan).
    def n_parametros(self):
        return sum(int(np.prod(peso.shape)) for peso in self.modelo.trainable_weights)
