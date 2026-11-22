# Modelo Seq2Seq (encoder-decoder con LSTM), con atención opcional.
#
# La base es la misma arquitectura de la primera versión del chatbot:
#   - Encoder: Embedding + LSTM. Lee la pregunta y la resume en dos estados (h y c).
#   - Decoder: Embedding + LSTM que empieza con esos estados, y una Dense con softmax
#     que da la probabilidad de cada palabra del vocabulario de respuestas.
#
# Sobre esa base se pueden activar dos mejoras:
#   - atencion=True: en cada paso el decoder vuelve a mirar todas las palabras de la
#     pregunta y decide en cuáles fijarse (atención de Luong, con producto escalar).
#   - matriz_embeddings: el encoder usa vectores preentrenados (congelados) en lugar de
#     aprender los suyos desde cero.

import numpy as np
from keras import Model, Input, layers, ops, initializers


def construir_seq2seq(n_encoder, n_decoder, max_encoder, max_decoder, emb_dim=64, unidades=128,
                      dropout=0.2, atencion=False, matriz_embeddings=None):
    # Devuelve tres modelos que comparten las mismas capas (y por tanto los mismos pesos):
    #   - modelo: el de entrenamiento (pregunta + respuesta desplazada -> siguiente palabra)
    #   - modelo_encoder: pregunta -> salidas y estados del encoder
    #   - modelo_paso: genera UNA palabra de la respuesta a partir de la anterior

    # ------------------------------------------------------------------ Encoder
    # Entrada del encoder: la pregunta ya convertida a índices y con padding.
    encoder_ids = Input(shape=(max_encoder,), dtype="int32", name="encoder_ids")

    # Embedding del encoder: preentrenado y congelado, o aprendido desde cero.
    # mask_zero=True hace que las posiciones de padding (índice 0) se ignoren.
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

    # LSTM del encoder. return_sequences=True devuelve la salida de cada palabra (la usa la
    # atención) y return_state=True los estados finales h y c (los usa el decoder).
    lstm_encoder = layers.LSTM(unidades, return_sequences=True, return_state=True,
                               dropout=dropout, name="lstm_encoder")
    salidas_encoder, estado_h, estado_c = lstm_encoder(emb_encoder(encoder_ids))

    # ------------------------------------------------------------------ Decoder
    # Entrada del decoder: la respuesta desplazada, que empieza por <start> (teacher forcing).
    decoder_ids = Input(shape=(max_decoder,), dtype="int32", name="decoder_ids")

    emb_decoder = layers.Embedding(n_decoder, emb_dim, mask_zero=True, name="emb_decoder")
    lstm_decoder = layers.LSTM(unidades, return_sequences=True, return_state=True,
                               dropout=dropout, name="lstm_decoder")

    # El decoder empieza con los estados finales del encoder.
    salidas_decoder, _, _ = lstm_decoder(emb_decoder(decoder_ids), initial_state=[estado_h, estado_c])

    # ------------------------------------------------------------- Capa final
    # Estas capas se crean una sola vez y se usan tanto al entrenar como al generar.
    if atencion:
        capa_atencion = layers.Attention(name="atencion")
        capa_combinar = layers.Dense(unidades, activation="tanh", name="combinar")
    else:
        capa_atencion = None
        capa_combinar = None
    capa_softmax = layers.Dense(n_decoder, activation="softmax", name="softmax")

    def capa_final(salida_decoder, salidas_enc, ids_enc):
        # Sin atención: la salida del decoder va directa a la softmax.
        if capa_atencion is None:
            return capa_softmax(salida_decoder), None

        # Con atención: el decoder compara su salida con la de cada palabra de la pregunta.
        # La máscara evita que mire las posiciones de padding.
        mascara_encoder = ops.not_equal(ids_enc, 0)
        contexto, pesos = capa_atencion([salida_decoder, salidas_enc],
                                        mask=[None, mascara_encoder],
                                        return_attention_scores=True)

        # Juntamos la salida del decoder con el contexto y lo pasamos por una Dense.
        mezcla = capa_combinar(layers.Concatenate()([salida_decoder, contexto]))
        return capa_softmax(mezcla), pesos

    # ------------------------------------------------- Modelo de entrenamiento
    probabilidades, _ = capa_final(salidas_decoder, salidas_encoder, encoder_ids)

    modelo = Model([encoder_ids, decoder_ids], probabilidades, name="seq2seq")
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # ---------------------------------------------------- Modelos de inferencia
    # Encoder de inferencia: pregunta -> salidas de cada palabra + estados finales.
    modelo_encoder = Model(encoder_ids, [salidas_encoder, estado_h, estado_c], name="encoder_inferencia")

    # Decoder de inferencia: recibe UN token y los estados del paso anterior, y devuelve
    # las probabilidades del siguiente token y los estados nuevos.
    token = Input(shape=(1,), dtype="int32")
    h_anterior = Input(shape=(unidades,))
    c_anterior = Input(shape=(unidades,))
    salidas_enc_entrada = Input(shape=(max_encoder, unidades))
    ids_enc_entrada = Input(shape=(max_encoder,), dtype="int32")

    salida, h_nuevo, c_nuevo = lstm_decoder(emb_decoder(token), initial_state=[h_anterior, c_anterior])
    probs_paso, pesos_paso = capa_final(salida, salidas_enc_entrada, ids_enc_entrada)

    if pesos_paso is None:
        modelo_paso = Model([token, h_anterior, c_anterior],
                            [probs_paso, h_nuevo, c_nuevo], name="decoder_paso")
    else:
        # Con atención, el decoder también necesita las salidas del encoder y devuelve los pesos.
        modelo_paso = Model([token, h_anterior, c_anterior, salidas_enc_entrada, ids_enc_entrada],
                            [probs_paso, h_nuevo, c_nuevo, pesos_paso], name="decoder_paso")

    return modelo, modelo_encoder, modelo_paso


def generar(modelo_encoder, modelo_paso, encoder_ids, id_start, id_end, max_decoder, atencion):
    # Genera la respuesta de muchas frases a la vez (decodificación voraz por lotes).
    #
    # En cada paso nos quedamos con la palabra más probable y la usamos como entrada del
    # paso siguiente, hasta llegar a <end>. Hacerlo por lotes es mucho más rápido que
    # generar las respuestas una a una.
    #
    # Devuelve los índices generados (frases x max_decoder) y, si hay atención, los pesos de
    # atención de cada paso (frases x max_decoder x max_encoder).

    # 1. El encoder resume todas las preguntas.
    encoder_ids = ops.convert_to_tensor(encoder_ids)
    salidas_enc, h, c = modelo_encoder(encoder_ids, training=False)

    n_frases = encoder_ids.shape[0]

    # 2. Todas las respuestas empiezan con <start>.
    token = ops.convert_to_tensor(np.full((n_frases, 1), id_start, dtype="int32"))

    generados = np.zeros((n_frases, max_decoder), dtype="int32")
    pesos = np.zeros((n_frases, max_decoder, encoder_ids.shape[1]), dtype="float32")
    terminado = np.zeros(n_frases, dtype=bool)

    # 3. Generamos palabra a palabra.
    for paso in range(max_decoder):
        if atencion:
            probs, h, c, p = modelo_paso([token, h, c, salidas_enc, encoder_ids], training=False)
            pesos[:, paso] = np.asarray(p)[:, 0]
        else:
            probs, h, c = modelo_paso([token, h, c], training=False)

        # Elegimos la palabra con mayor probabilidad.
        siguiente = np.argmax(np.asarray(probs)[:, -1], axis=-1).astype("int32")

        # Las frases que ya terminaron solo añaden relleno.
        siguiente[terminado] = 0
        generados[:, paso] = siguiente

        # Una frase termina al generar <end> (o relleno).
        terminado |= (siguiente == id_end) | (siguiente == 0)
        if terminado.all():
            break

        # La palabra generada es la entrada del siguiente paso.
        token = ops.convert_to_tensor(siguiente[:, None])

    return generados, pesos
