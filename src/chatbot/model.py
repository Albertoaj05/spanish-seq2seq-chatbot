"""Modelo Seq2Seq (encoder-decoder con LSTM), con atención opcional.

La base es la misma arquitectura de la primera versión: Embedding + LSTM en el encoder,
Embedding + LSTM en el decoder y una capa Dense con softmax sobre el vocabulario de
respuestas. Sobre ella se pueden activar dos mejoras:

- atencion=True: en cada paso el decoder mira todas las salidas del encoder y decide en
  qué palabras de la pregunta fijarse (atención de Luong, producto escalar).
- matriz_embeddings: el encoder usa vectores preentrenados congelados en lugar de
  aprender los suyos desde cero.
"""

import numpy as np
from keras import Model, Input, layers, ops, initializers


def construir_seq2seq(n_encoder, n_decoder, max_encoder, max_decoder, emb_dim=64, unidades=128,
                      dropout=0.2, atencion=False, matriz_embeddings=None):
    # ----- Encoder -----
    encoder_ids = Input(shape=(max_encoder,), dtype="int32", name="encoder_ids")
    if matriz_embeddings is not None:
        emb_encoder = layers.Embedding(
            n_encoder, matriz_embeddings.shape[1], mask_zero=True, trainable=False,
            embeddings_initializer=initializers.Constant(matriz_embeddings), name="emb_encoder")
    else:
        emb_encoder = layers.Embedding(n_encoder, emb_dim, mask_zero=True, name="emb_encoder")
    lstm_encoder = layers.LSTM(unidades, return_sequences=True, return_state=True,
                               dropout=dropout, name="lstm_encoder")
    salidas_encoder, h, c = lstm_encoder(emb_encoder(encoder_ids))

    # ----- Decoder -----
    decoder_ids = Input(shape=(max_decoder,), dtype="int32", name="decoder_ids")
    emb_decoder = layers.Embedding(n_decoder, emb_dim, mask_zero=True, name="emb_decoder")
    lstm_decoder = layers.LSTM(unidades, return_sequences=True, return_state=True,
                               dropout=dropout, name="lstm_decoder")
    salidas_decoder, _, _ = lstm_decoder(emb_decoder(decoder_ids), initial_state=[h, c])

    capa_atencion = layers.Attention(name="atencion") if atencion else None
    combinar = layers.Dense(unidades, activation="tanh", name="combinar") if atencion else None
    softmax = layers.Dense(n_decoder, activation="softmax", name="softmax")

    def cabeza(salida_decoder, salidas_enc, ids_enc):
        """Capa final, compartida entre el modelo de entrenamiento y el de inferencia."""
        if capa_atencion is None:
            return softmax(salida_decoder), None
        mascara_encoder = ops.not_equal(ids_enc, 0)
        contexto, pesos = capa_atencion([salida_decoder, salidas_enc], mask=[None, mascara_encoder],
                                        return_attention_scores=True)
        mezcla = combinar(layers.Concatenate()([salida_decoder, contexto]))
        return softmax(mezcla), pesos

    probabilidades, _ = cabeza(salidas_decoder, salidas_encoder, encoder_ids)
    modelo = Model([encoder_ids, decoder_ids], probabilidades, name="seq2seq")
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # ----- Modelos de inferencia -----
    # Encoder: pregunta -> salidas de cada palabra + estados finales.
    modelo_encoder = Model(encoder_ids, [salidas_encoder, h, c], name="encoder_inferencia")

    # Decoder paso a paso: un token + estados anteriores -> probabilidades del siguiente token.
    token = Input(shape=(1,), dtype="int32")
    h_in, c_in = Input(shape=(unidades,)), Input(shape=(unidades,))
    salidas_enc_in = Input(shape=(max_encoder, unidades))
    ids_enc_in = Input(shape=(max_encoder,), dtype="int32")
    salida, h_out, c_out = lstm_decoder(emb_decoder(token), initial_state=[h_in, c_in])
    probs_paso, pesos_paso = cabeza(salida, salidas_enc_in, ids_enc_in)
    if pesos_paso is None:
        modelo_paso = Model([token, h_in, c_in], [probs_paso, h_out, c_out], name="decoder_paso")
    else:
        modelo_paso = Model([token, h_in, c_in, salidas_enc_in, ids_enc_in],
                            [probs_paso, h_out, c_out, pesos_paso], name="decoder_paso")

    return modelo, modelo_encoder, modelo_paso


def generar(modelo_encoder, modelo_paso, encoder_ids, id_start, id_end, max_decoder, atencion):
    """Decodificación voraz por lotes: genera la respuesta de muchas frases a la vez.

    Devuelve los índices generados (lote x max_decoder) y, si hay atención, los pesos de
    atención de cada paso (lote x max_decoder x max_encoder).
    """
    encoder_ids = ops.convert_to_tensor(encoder_ids)
    salidas_enc, h, c = modelo_encoder(encoder_ids, training=False)
    n = encoder_ids.shape[0]
    token = ops.convert_to_tensor(np.full((n, 1), id_start, dtype="int32"))
    generados = np.zeros((n, max_decoder), dtype="int32")
    pesos = np.zeros((n, max_decoder, encoder_ids.shape[1]), dtype="float32")
    terminado = np.zeros(n, dtype=bool)

    for paso in range(max_decoder):
        if atencion:
            probs, h, c, p = modelo_paso([token, h, c, salidas_enc, encoder_ids], training=False)
            pesos[:, paso] = np.asarray(p)[:, 0]
        else:
            probs, h, c = modelo_paso([token, h, c], training=False)
        siguiente = np.argmax(np.asarray(probs)[:, -1], axis=-1).astype("int32")
        siguiente[terminado] = 0
        generados[:, paso] = siguiente
        terminado |= (siguiente == id_end) | (siguiente == 0)
        if terminado.all():
            break
        token = ops.convert_to_tensor(siguiente[:, None])
    return generados, pesos
