# Gráficas del README.
#
#     python -m chatbot.figures
#
# figures/results.png: aciertos de cada versión en test y hard (a partir de results/metrics.csv).
# figures/attention.png: mapa de atención de la versión final en dos preguntas.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bot import Chatbot
from .data import aumentar, cargar_intenciones, pares, respuestas_validas
from .text import tokenizar

# Carpetas del proyecto.
RAIZ = Path(__file__).resolve().parents[2]
RESULTADOS = RAIZ / "results"
FIGURAS = RAIZ / "figures"

# Colores de las gráficas.
AZUL = "#2a78d6"       # test
NARANJA = "#eb6834"    # hard
TINTA = "#0b0b0b"      # texto principal
TINTA_2 = "#52514e"    # texto secundario y ejes
REJILLA = "#e4e3df"
FONDO = "#fcfcfb"

# Nombre de cada versión en el eje de la gráfica.
NOMBRES = {
    "seq2seq": "Seq2Seq",
    "seq2seq_attention": "+ attention",
    "seq2seq_pretrained": "+ pretrained\nembeddings",
    "seq2seq_attention_pretrained": "+ attention\n+ embeddings",
}


# Dejamos los ejes limpios: sin bordes arriba ni a los lados y con una rejilla suave.
def estilo(ax):
    ax.set_facecolor(FONDO)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, length=0)
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


# Barras con la media de las 5 semillas de cada versión y su desviación típica.
def figura_resultados():
    metricas = pd.read_csv(RESULTADOS / "metrics.csv")

    # Media y desviación de cada versión del chatbot.
    generativas = metricas[metricas.version != "retrieval_baseline"]
    resumen = generativas.groupby("version", sort=False)[["acierto_test", "acierto_hard"]].agg(["mean", "std"])

    # La referencia de recuperación solo tiene una ejecución.
    recuperacion = metricas[metricas.version == "retrieval_baseline"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(FONDO)
    estilo(ax)

    x = np.arange(len(resumen))
    ancho = 0.36
    grupos = [("acierto_test", AZUL, "Test: new phrasings (72)"),
              ("acierto_hard", NARANJA, "Hard: no shared words (48)")]

    # Dibujamos una barra para test y otra para hard en cada versión.
    for i, (columna, color, etiqueta) in enumerate(grupos):
        medias = resumen[(columna, "mean")].values * 100
        desviaciones = resumen[(columna, "std")].values * 100
        posiciones = x + (i - 0.5) * (ancho + 0.02)

        ax.bar(posiciones, medias, ancho, color=color, label=etiqueta, zorder=2)
        ax.errorbar(posiciones, medias, yerr=desviaciones, fmt="none",
                    ecolor=TINTA_2, elinewidth=1, capsize=3, zorder=3)

        # Escribimos el porcentaje encima de cada barra.
        for posicion, media, desviacion in zip(posiciones, medias, desviaciones):
            ax.text(posicion, media + desviacion + 1.5, f"{media:.0f}%",
                    ha="center", va="bottom", fontsize=9, color=TINTA)

    # La referencia de recuperación se dibuja como líneas horizontales.
    for columna, estilo_linea, nombre in [("acierto_test", "--", "test"), ("acierto_hard", ":", "hard")]:
        ax.axhline(recuperacion[columna] * 100, color=TINTA_2, linestyle=estilo_linea, linewidth=1,
                   label=f"Retrieval baseline, {nombre} ({recuperacion[columna] * 100:.0f}%)")

    # Ejes, leyenda y título.
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


# Mapa de atención: en qué palabras de la pregunta se fija el decoder al escribir cada palabra.
def figura_atencion():
    # Entrenamos la versión final.
    intenciones = cargar_intenciones()
    bot = Chatbot(atencion=True, preentrenado=True, semilla=0)
    bot.entrenar(aumentar(pares(intenciones, "train")), pares(intenciones, "val"))

    # Probamos varias preguntas y nos quedamos con las dos primeras que responde bien.
    validas = respuestas_validas(intenciones)
    candidatas = [("Oye, ¿qué hace exactamente el decoder?", "decoder"),
                  ("Me explicas el sobreajuste por favor", "sobreajuste"),
                  ("Perdona, ¿qué es una LSTM?", "lstm"),
                  ("¿Hay tormenta hoy?", "tiempo"),
                  ("Me explicas la atención por favor", "atencion")]

    respuestas, pesos = bot.responder_lote([frase for frase, _ in candidatas], devolver_atencion=True)

    elegidas = []
    for i, (respuesta, (_, intencion)) in enumerate(zip(respuestas, candidatas)):
        if respuesta in validas[intencion]:
            elegidas.append(i)
    elegidas = elegidas[:2]

    frases = [candidatas[i][0] for i in elegidas]
    respuestas = [respuestas[i] for i in elegidas]
    pesos = pesos[elegidas]

    # Una matriz por pregunta. Filas: palabras de la respuesta. Columnas: palabras de la pregunta.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200,
                             gridspec_kw={"width_ratios": [len(tokenizar(f)) for f in frases]})
    fig.patch.set_facecolor(FONDO)

    for ax, frase, respuesta, p in zip(axes, frases, respuestas, pesos):
        palabras_pregunta = tokenizar(frase)
        palabras_respuesta = respuesta.split()
        matriz = p[:len(palabras_respuesta), :len(palabras_pregunta)]

        ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(palabras_pregunta)), palabras_pregunta,
                      rotation=35, ha="right", fontsize=9, color=TINTA)
        ax.set_yticks(range(len(palabras_respuesta)), palabras_respuesta, fontsize=9, color=TINTA)
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
