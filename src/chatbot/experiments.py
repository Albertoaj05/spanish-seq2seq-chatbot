# Comparación de las cuatro versiones del chatbot, entrenando cada una con 5 semillas.
#
#     python -m chatbot.experiments            # 5 semillas por versión (unos 20 min en CPU)
#     python -m chatbot.experiments --seeds 2  # prueba rápida
#
# Los resultados se guardan en results/: metrics.csv (aciertos de cada entrenamiento),
# predictions.csv (cada respuesta generada) e histories.json (curvas de entrenamiento).

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .bot import Chatbot
from .data import aumentar, cargar_intenciones, pares, respuestas_validas
from .retrieval import Recuperacion

RESULTADOS = Path(__file__).resolve().parents[2] / "results"

# Versiones que comparamos.
VERSIONES = {
    "seq2seq": {"atencion": False, "preentrenado": False},
    "seq2seq_attention": {"atencion": True, "preentrenado": False},
    "seq2seq_pretrained": {"atencion": False, "preentrenado": True},
    "seq2seq_attention_pretrained": {"atencion": True, "preentrenado": True},
}


def evaluar(modelo, conjuntos, validas):
    # Una respuesta es correcta si coincide con alguna respuesta válida de su intención.
    filas = []

    for grupo, lista in conjuntos.items():
        generadas = modelo.responder_lote([frase for frase, _, _ in lista])

        for (frase, _, intencion), generada in zip(lista, generadas):
            filas.append({
                "split": grupo,
                "frase": frase,
                "intencion": intencion,
                "respuesta": generada,
                "correcta": generada in validas[intencion],
            })

    return filas


def resumen_aciertos(filas, conjuntos):
    # Porcentaje de aciertos en cada grupo.
    df = pd.DataFrame(filas)
    return {f"acierto_{grupo}": df[df.split == grupo].correcta.mean() for grupo in conjuntos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    # Cargamos los datos: entrenamiento con aumento y los tres grupos de evaluación.
    intenciones = cargar_intenciones()
    validas = respuestas_validas(intenciones)
    train = aumentar(pares(intenciones, "train"), n_copias=3)
    conjuntos = {grupo: pares(intenciones, grupo) for grupo in ["val", "test", "hard"]}

    metricas = []
    predicciones = []
    historiales = {}
    RESULTADOS.mkdir(exist_ok=True)

    # Referencia de recuperación. No depende de ninguna semilla, así que basta con una ejecución.
    inicio = time.time()
    recuperacion = Recuperacion().entrenar(pares(intenciones, "train"))
    filas = evaluar(recuperacion, conjuntos, validas)
    for fila in filas:
        fila.update(version="retrieval_baseline", semilla=0)
    predicciones += filas
    metricas.append({"version": "retrieval_baseline", "semilla": 0, "epocas": 0, "parametros": 0,
                     "segundos": round(time.time() - inicio, 1), **resumen_aciertos(filas, conjuntos)})

    # Entrenamos y evaluamos cada versión con varias semillas.
    for nombre, configuracion in VERSIONES.items():
        for semilla in range(args.seeds):
            inicio = time.time()
            bot = Chatbot(semilla=semilla, **configuracion).entrenar(train, conjuntos["val"])

            filas = evaluar(bot, conjuntos, validas)
            for fila in filas:
                fila.update(version=nombre, semilla=semilla)
            predicciones += filas

            resultado = {"version": nombre, "semilla": semilla,
                         "epocas": len(bot.historial["loss"]), "parametros": bot.n_parametros(),
                         "segundos": round(time.time() - inicio, 1), **resumen_aciertos(filas, conjuntos)}
            metricas.append(resultado)
            historiales[f"{nombre}/{semilla}"] = {k: [float(x) for x in v] for k, v in bot.historial.items()}

            print(f"{nombre:30} semilla {semilla}  "
                  f"val {resultado['acierto_val']:.0%}  test {resultado['acierto_test']:.0%}  "
                  f"hard {resultado['acierto_hard']:.0%}  ({resultado['epocas']} épocas)", flush=True)

            # Guardamos después de cada entrenamiento por si se corta a mitad.
            pd.DataFrame(metricas).to_csv(RESULTADOS / "metrics.csv", index=False)
            pd.DataFrame(predicciones).to_csv(RESULTADOS / "predictions.csv", index=False)
            with open(RESULTADOS / "histories.json", "w") as fichero:
                json.dump(historiales, fichero)


if __name__ == "__main__":
    main()
