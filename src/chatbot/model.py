# Modelo Seq2Seq con LSTM, con atención opcional.
# Es la misma arquitectura de la primera versión del chatbot (Embedding + LSTM en el encoder
# y en el decoder), a la que se le pueden añadir la atención y los embeddings preentrenados.

import numpy as np
from keras import Model, Input, layers, ops, initializers


# Construye el modelo de entrenamiento y los dos modelos de inferencia (encoder y decoder).
# Los tres comparten las mismas capas, así que comparten también los pesos.
def construir_seq2seq(n_encoder, n_decoder, max_encoder, max_decoder, emb_dim=64, unidades=128,
                      dropout=0.2, atencion=False, matriz_embeddings=None):

    # Entrada del encoder: frases convertidas a índices y con padding.
    encoder_ids = Input(shape=(max_encoder,), dtype="int32", name="encoder_ids")

    # Embedding del encoder. Con los vectores de spaCy lo dejamos congelado (trainable=False).
    # mask_zero=True permite ignorar las posiciones de padding.
    if matriz_embeddings is not None:
        emb_encoder = layers.Embedding(
            n_encoder, matriz_embeddings.shape[1],
            mask_zero=True,
            trainable=False,
            embeddings_initializer=initializers.Constant(matriz_embeddings),
            name="emb_encoder",
        )
    else:
        emb_encoder = layers.Embedding(n_encoder, emb_dim, mask_zero=True, name="emb_encoder")

    # LSTM del encoder. Devuelve la salida de cada palabra (la usa la atención)
    # y los estados finales h y c (los usa el decoder para empezar).
    lstm_encoder = layers.LSTM(unidades, return_sequences=True, return_state=True,
                               dropout=dropout, name="lstm_encoder")
    salidas_encoder, estado_h, estado_c = lstm_encoder(emb_encoder(encoder_ids))

    # Entrada del decoder: respuestas desplazadas que empiezan con <start>.
    decoder_ids = Input(shape=(max_decoder,), dtype="int32", name="decoder_ids")

    # Embedding y LSTM del decoder.
    emb_decoder = layers.Embedding(n_decoder, emb_dim, mask_zero=True, name="emb_decoder")
    lstm_decoder = layers.LSTM(unidades, return_sequences=True, return_state=True,
                               dropout=dropout, name="lstm_decoder")

    # El decoder empieza con los estados finales del encoder.
    salidas_decoder, _, _ = lstm_decoder(emb_decoder(decoder_ids), initial_state=[estado_h, estado_c])

    # Capas de la parte final. Se crean una sola vez y se usan al entrenar y al generar.
    if atencion:
        capa_atencion = layers.Attention(name="atencion")
        capa_combinar = layers.Dense(unidades, activation="tanh", name="combinar")
    else:
        capa_atencion = None
        capa_combinar = None
    capa_softmax = layers.Dense(n_decoder, activation="softmax", name="softmax")

    def capa_final(salida_decoder, salidas_enc, ids_enc):
        # Sin atención, la salida del decoder va directamente a la softmax.
        if capa_atencion is None:
            return capa_softmax(salida_decoder), None

        # Con atención, el decoder mira todas las palabras de la pregunta.
        # La máscara evita que se fije en el padding.
        mascara_encoder = ops.not_equal(ids_enc, 0)
        contexto, pesos = capa_atencion([salida_decoder, salidas_enc],
                                        mask=[None, mascara_encoder],
                                        return_attention_scores=True)

        # Juntamos la salida del decoder con el contexto antes de la softmax.
        mezcla = capa_combinar(layers.Concatenate()([salida_decoder, contexto]))
        return capa_softmax(mezcla), pesos

    # Modelo completo de entrenamiento.
    probabilidades, _ = capa_final(salidas_decoder, salidas_encoder, encoder_ids)

    modelo = Model([encoder_ids, decoder_ids], probabilidades, name="seq2seq")
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Modelo encoder de inferencia: recibe la frase y devuelve sus salidas y estados.
    modelo_encoder = Model(encoder_ids, [salidas_encoder, estado_h, estado_c], name="encoder_inferencia")

    # Entradas del decoder de inferencia: un solo token y los estados del paso anterior.
    token = Input(shape=(1,), dtype="int32")
    h_anterior = Input(shape=(unidades,))
    c_anterior = Input(shape=(unidades,))
    salidas_enc_entrada = Input(shape=(max_encoder, unidades))
    ids_enc_entrada = Input(shape=(max_encoder,), dtype="int32")

    salida, h_nuevo, c_nuevo = lstm_decoder(emb_decoder(token), initial_state=[h_anterior, c_anterior])
    probs_paso, pesos_paso = capa_final(salida, salidas_enc_entrada, ids_enc_entrada)

    # Modelo decoder de inferencia. Con atención necesita además las salidas del encoder.
    if pesos_paso is None:
        modelo_paso = Model([token, h_anterior, c_anterior],
                            [probs_paso, h_nuevo, c_nuevo], name="decoder_paso")
    else:
        modelo_paso = Model([token, h_anterior, c_anterior, salidas_enc_entrada, ids_enc_entrada],
                            [probs_paso, h_nuevo, c_nuevo, pesos_paso], name="decoder_paso")

    return modelo, modelo_encoder, modelo_paso


# Genera las respuestas de varias frases a la vez, palabra a palabra.
# Devuelve los índices generados y, si hay atención, los pesos de cada paso.
def generar(modelo_encoder, modelo_paso, encoder_ids, id_start, id_end, max_decoder, atencion):

    # El encoder obtiene las salidas y los estados de todas las frases.
    encoder_ids = ops.convert_to_tensor(encoder_ids)
    salidas_enc, h, c = modelo_encoder(encoder_ids, training=False)

    n_frases = encoder_ids.shape[0]

    # La generación empieza con el token <start>.
    token = ops.convert_to_tensor(np.full((n_frases, 1), id_start, dtype="int32"))

    generados = np.zeros((n_frases, max_decoder), dtype="int32")
    pesos = np.zeros((n_frases, max_decoder, encoder_ids.shape[1]), dtype="float32")
    terminado = np.zeros(n_frases, dtype=bool)

    # Generamos la respuesta token a token.
    for paso in range(max_decoder):
        if atencion:
            probs, h, c, p = modelo_paso([token, h, c, salidas_enc, encoder_ids], training=False)
            pesos[:, paso] = np.asarray(p)[:, 0]
        else:
            probs, h, c = modelo_paso([token, h, c], training=False)

        # Seleccionamos el token con mayor probabilidad.
        siguiente = np.argmax(np.asarray(probs)[:, -1], axis=-1).astype("int32")

        # Las frases que ya han terminado solo añaden relleno.
        siguiente[terminado] = 0
        generados[:, paso] = siguiente

        # Si aparece <end>, se termina la generación de esa frase.
        terminado |= (siguiente == id_end) | (siguiente == 0)
        if terminado.all():
            break

        # El token predicho se usa como entrada en el siguiente paso.
        token = ops.convert_to_tensor(siguiente[:, None])

    return generados, pesos
