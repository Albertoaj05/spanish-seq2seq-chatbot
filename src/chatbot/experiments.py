"""Comparación de las cuatro versiones del chatbot con varias semillas.

Uso:
    python -m chatbot.experiments            # 5 semillas por versión
    python -m chatbot.experiments --seeds 2  # prueba rápida

Guarda en results/:
    metrics.csv      una fila por versión y semilla (aciertos, épocas, tiempo...)
    predictions.csv  la respuesta generada para cada frase de val, test y hard
    histories.json   curvas de pérdida de entrenamiento y validación
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .bot import Chatbot
from .data import aumentar, cargar_intenciones, pares, respuestas_validas
from .retrieval import Recuperacion

RESULTADOS = Path(__file__).resolve().parents[2] / "results"

VERSIONES = {
    "seq2seq": dict(atencion=False, preentrenado=False),
    "seq2seq_attention": dict(atencion=True, preentrenado=False),
    "seq2seq_pretrained": dict(atencion=False, preentrenado=True),
    "seq2seq_attention_pretrained": dict(atencion=True, preentrenado=True),
}


def evaluar(modelo, conjuntos, validas):
    """Una respuesta es correcta si coincide con alguna respuesta válida de su intención."""
    filas = []
    for split, lista in conjuntos.items():
        generadas = modelo.responder_lote([frase for frase, _, _ in lista])
        for (frase, _, intencion), generada in zip(lista, generadas):
            filas.append(dict(split=split, frase=frase, intencion=intencion,
                              respuesta=generada, correcta=generada in validas[intencion]))
    return filas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    intenciones = cargar_intenciones()
    validas = respuestas_validas(intenciones)
    train = aumentar(pares(intenciones, "train"), n_copias=3)
    conjuntos = {s: pares(intenciones, s) for s in ["val", "test", "hard"]}

    metricas, predicciones, historiales = [], [], {}
    RESULTADOS.mkdir(exist_ok=True)

    # Referencia no generativa (determinista, una sola ejecución).
    t0 = time.time()
    recuperacion = Recuperacion().entrenar(pares(intenciones, "train"))
    filas = evaluar(recuperacion, conjuntos, validas)
    for f in filas:
        f.update(version="retrieval_baseline", semilla=0)
    predicciones += filas
    df = pd.DataFrame(filas)
    metricas.append(dict(version="retrieval_baseline", semilla=0, epocas=0, parametros=0,
                         segundos=round(time.time() - t0, 1),
                         **{f"acierto_{s}": df[df.split == s].correcta.mean() for s in conjuntos}))

    for nombre, config in VERSIONES.items():
        for semilla in range(args.seeds):
            t0 = time.time()
            bot = Chatbot(semilla=semilla, **config).entrenar(train, conjuntos["val"])
            filas = evaluar(bot, conjuntos, validas)
            for f in filas:
                f.update(version=nombre, semilla=semilla)
            predicciones += filas
            df = pd.DataFrame(filas)
            fila = dict(version=nombre, semilla=semilla, epocas=len(bot.historial["loss"]),
                        parametros=bot.n_parametros(), segundos=round(time.time() - t0, 1),
                        **{f"acierto_{s}": df[df.split == s].correcta.mean() for s in conjuntos})
            metricas.append(fila)
            historiales[f"{nombre}/{semilla}"] = {k: [float(x) for x in v]
                                                  for k, v in bot.historial.items()}
            print(fila, flush=True)

            # Guardamos después de cada ejecución por si se corta a mitad.
            pd.DataFrame(metricas).to_csv(RESULTADOS / "metrics.csv", index=False)
            pd.DataFrame(predicciones).to_csv(RESULTADOS / "predictions.csv", index=False)
            with open(RESULTADOS / "histories.json", "w") as f:
                json.dump(historiales, f)


if __name__ == "__main__":
    main()
