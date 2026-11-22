# Chat con el bot desde la terminal.
#
#     python -m chatbot.chat                      # versión final (atención + embeddings)
#     python -m chatbot.chat --version seq2seq    # versión base
#
# La primera vez entrena el modelo y lo guarda en models/, así después arranca directamente.

import argparse
import pickle
from pathlib import Path

from .bot import Chatbot
from .data import aumentar, cargar_intenciones, pares
from .experiments import VERSIONES
from .model import construir_seq2seq

MODELOS = Path(__file__).resolve().parents[2] / "models"


def cargar_o_entrenar(version, semilla=0):
    carpeta = MODELOS / version
    bot = Chatbot(semilla=semilla, **VERSIONES[version])

    # Si el modelo ya está guardado, cargamos los vocabularios y los pesos.
    if (carpeta / "estado.pkl").exists():
        with open(carpeta / "estado.pkl", "rb") as fichero:
            bot.__dict__.update(pickle.load(fichero))

        matriz = bot.vocab_enc.matriz if bot.preentrenado else None
        bot.modelo, bot.modelo_encoder, bot.modelo_paso = construir_seq2seq(
            len(bot.vocab_enc), len(bot.vocab_dec), bot.max_encoder, bot.max_decoder,
            emb_dim=bot.emb_dim, unidades=bot.unidades, dropout=bot.dropout,
            atencion=bot.atencion, matriz_embeddings=matriz,
        )
        bot.modelo.load_weights(carpeta / "pesos.weights.h5")
        return bot

    # Si no, lo entrenamos y lo guardamos para la próxima vez.
    print("Entrenando el modelo (solo la primera vez)...")
    intenciones = cargar_intenciones()
    bot.entrenar(aumentar(pares(intenciones, "train")), pares(intenciones, "val"))

    carpeta.mkdir(parents=True, exist_ok=True)
    bot.modelo.save_weights(carpeta / "pesos.weights.h5")

    # Guardamos todo menos los modelos de Keras, que se vuelven a construir al cargar.
    estado = {}
    for clave, valor in bot.__dict__.items():
        if clave not in ("modelo", "modelo_encoder", "modelo_paso", "historial"):
            estado[clave] = valor
    with open(carpeta / "estado.pkl", "wb") as fichero:
        pickle.dump(estado, fichero)

    return bot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="seq2seq_attention_pretrained", choices=list(VERSIONES))
    args = parser.parse_args()

    bot = cargar_o_entrenar(args.version)
    print("Escribe una frase (o 'salir' para terminar).\n")

    while True:
        try:
            frase = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not frase or frase.lower() == "salir":
            break

        print("Bot:", bot.responder(frase), "\n")


if __name__ == "__main__":
    main()
