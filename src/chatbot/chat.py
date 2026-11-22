"""Conversación con el chatbot desde la terminal.

Uso:
    python -m chatbot.chat                      # versión final (atención + embeddings)
    python -m chatbot.chat --version seq2seq    # versión base

La primera vez entrena el modelo (menos de un minuto en CPU) y guarda los pesos en
models/ para no tener que repetirlo.
"""

import argparse
import pickle
from pathlib import Path

from .bot import Chatbot
from .data import aumentar, cargar_intenciones, pares
from .experiments import VERSIONES
from .model import construir_seq2seq

MODELOS = Path(__file__).resolve().parents[2] / "models"


def cargar_o_entrenar(version, semilla=0):
    ruta = MODELOS / version
    bot = Chatbot(semilla=semilla, **VERSIONES[version])
    if (ruta / "estado.pkl").exists():
        with open(ruta / "estado.pkl", "rb") as f:
            estado = pickle.load(f)
        bot.__dict__.update(estado)
        matriz = bot.vocab_enc.matriz if bot.preentrenado else None
        bot.modelo, bot.modelo_encoder, bot.modelo_paso = construir_seq2seq(
            len(bot.vocab_enc), len(bot.vocab_dec), bot.max_encoder, bot.max_decoder,
            emb_dim=bot.emb_dim, unidades=bot.unidades, dropout=bot.dropout,
            atencion=bot.atencion, matriz_embeddings=matriz)
        bot.modelo.load_weights(ruta / "pesos.weights.h5")
        return bot

    print("Entrenando el modelo (solo la primera vez)...")
    intenciones = cargar_intenciones()
    bot.entrenar(aumentar(pares(intenciones, "train")), pares(intenciones, "val"))
    ruta.mkdir(parents=True, exist_ok=True)
    bot.modelo.save_weights(ruta / "pesos.weights.h5")
    estado = {k: v for k, v in bot.__dict__.items()
              if k not in ("modelo", "modelo_encoder", "modelo_paso", "historial")}
    with open(ruta / "estado.pkl", "wb") as f:
        pickle.dump(estado, f)
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
