"""Genera las figuras del README a partir de results/ (python -m chatbot.figures)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bot import Chatbot
from .data import aumentar, cargar_intenciones, pares
from .text import tokenizar

RAIZ = Path(__file__).resolve().parents[2]
RESULTADOS, FIGURAS = RAIZ / "results", RAIZ / "figures"

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

NOMBRES = {
    "seq2seq": "Seq2Seq",
    "seq2seq_attention": "+ attention",
    "seq2seq_pretrained": "+ pretrained\nembeddings",
    "seq2seq_attention_pretrained": "+ attention\n+ embeddings",
}


def estilo(ax):
    ax.set_facecolor(FONDO)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, length=0)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_resultados():
    m = pd.read_csv(RESULTADOS / "metrics.csv")
    gen = m[m.version != "retrieval_baseline"]
    resumen = gen.groupby("version", sort=False)[["acierto_test", "acierto_hard"]].agg(["mean", "std"])
    recuperacion = m[m.version == "retrieval_baseline"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(FONDO)
    estilo(ax)
    x = np.arange(len(resumen))
    ancho = 0.36
    for i, (col, color, etiqueta) in enumerate([("acierto_test", AZUL, "Test: new phrasings (72)"),
                                                ("acierto_hard", NARANJA, "Hard: no shared words (48)")]):
        medias = resumen[(col, "mean")].values * 100
        desv = resumen[(col, "std")].values * 100
        pos = x + (i - 0.5) * (ancho + 0.02)
        ax.bar(pos, medias, ancho, color=color, label=etiqueta, zorder=2)
        ax.errorbar(pos, medias, yerr=desv, fmt="none", ecolor=TINTA_2, elinewidth=1, capsize=3, zorder=3)
        for p, v, d in zip(pos, medias, desv):
            ax.text(p, v + d + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=TINTA)
    for col, estilo_linea, nombre in [("acierto_test", "--", "test"), ("acierto_hard", ":", "hard")]:
        ax.axhline(recuperacion[col] * 100, color=TINTA_2, linestyle=estilo_linea, linewidth=1,
                   label=f"Retrieval baseline, {nombre} ({recuperacion[col] * 100:.0f}%)")

    ax.set_xticks(x, [NOMBRES[v] for v in resumen.index], color=TINTA, fontsize=10)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("Correct answers", color=TINTA_2)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=TINTA, ncol=2)
    ax.set_title("Correct answers on sentences never seen in training (mean ± std, 5 seeds)",
                 loc="left", fontsize=11, color=TINTA, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURAS / "results.png", facecolor=FONDO)
    plt.close(fig)


def figura_atencion():
    intenciones = cargar_intenciones()
    bot = Chatbot(atencion=True, preentrenado=True, semilla=0)
    bot.entrenar(aumentar(pares(intenciones, "train")), pares(intenciones, "val"))
    from .data import respuestas_validas
    validas = respuestas_validas(intenciones)
    candidatas = [("Oye, ¿qué hace exactamente el decoder?", "decoder"),
                  ("Me explicas el sobreajuste por favor", "sobreajuste"),
                  ("Perdona, ¿qué es una LSTM?", "lstm"),
                  ("¿Hay tormenta hoy?", "tiempo"),
                  ("Me explicas la atención por favor", "atencion")]
    respuestas, pesos = bot.responder_lote([f for f, _ in candidatas], devolver_atencion=True)
    elegidas = [i for i, (r, (_, intencion)) in enumerate(zip(respuestas, candidatas))
                if r in validas[intencion]][:2]
    frases = [candidatas[i][0] for i in elegidas]
    respuestas = [respuestas[i] for i in elegidas]
    pesos = pesos[elegidas]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200,
                             gridspec_kw={"width_ratios": [len(tokenizar(f)) for f in frases]})
    fig.patch.set_facecolor(FONDO)
    for ax, frase, respuesta, p in zip(axes, frases, respuestas, pesos):
        entrada = tokenizar(frase)
        salida = respuesta.split()
        matriz = p[:len(salida), :len(entrada)]
        ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(entrada)), entrada, rotation=35, ha="right", fontsize=9, color=TINTA)
        ax.set_yticks(range(len(salida)), salida, fontsize=9, color=TINTA)
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(f"“{frase}”", fontsize=9.5, color=TINTA_2, pad=8)
    axes[0].set_ylabel("Generated answer", color=TINTA_2)
    fig.suptitle("Where the decoder looks while writing each word (darker = more attention)",
                 x=0.02, ha="left", fontsize=11, color=TINTA)
    fig.tight_layout()
    fig.savefig(FIGURAS / "attention.png", facecolor=FONDO)
    plt.close(fig)


def main():
    FIGURAS.mkdir(exist_ok=True)
    figura_resultados()
    figura_atencion()


if __name__ == "__main__":
    main()
