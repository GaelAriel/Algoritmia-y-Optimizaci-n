import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

import pandas as pd

import config
from caracteristicas import caracteristicas_extra
from tsp_algoritmos import (hill_climbing_2opt, iteraciones_sa, matriz_distancias,
                            mst_2opt, recocido_simulado_2opt)

COLUMNAS_COSTO = [config.COLUMNAS_COSTO[a] for a in config.ALGORITMOS]


def cargar_puntos(ruta_datos, instance_id):
    ruta = os.path.join(ruta_datos, "instances", f"{instance_id}.csv")
    with open(ruta, newline="") as f:
        return [(float(fila["x"]), float(fila["y"])) for fila in csv.DictReader(f)]


def parametros_experimento():
    return {
        "semilla_base": config.SEMILLA_BASE,
        "algoritmos": config.ALGORITMOS,
        "2opt_hill_climbing": {
            "tour_inicial": "aleatorio",
            "variante": "2-opt de primera mejora hasta optimo local (posicion 0 fija)",
            "reinicios": 1,
        },
        "mst_2opt": {
            "tour_inicial": "preorden del arbol de expansion minima (Prim desde la ciudad 0)",
            "mejora": "2-opt de primera mejora hasta optimo local",
        },
        "sa_2opt": {
            "tour_inicial": "aleatorio",
            "movimiento": "2-opt aleatorio (inversion de un tramo)",
            "aceptacion": "exp(-delta/T) si el tour empeora",
            "enfriamiento": "geometrico: T_t = T0 * (Tf/T0)^(t/N)",
            "T0": config.SA_T0,
            "Tf": config.SA_TF,
            "iteraciones": f"N = max({config.SA_MIN_ITER}, {config.SA_FACTOR_ITER} * n^2)",
            "costo_reportado": "longitud del mejor tour visto durante la corrida",
        },
        "regla_de_empate": {
            "tolerancia": config.TOLERANCIA_EMPATE,
            "prioridad": config.ALGORITMOS,
        },
        "esquema_de_semillas": "semilla = SEMILLA_BASE * 1000000 + numero_de_instancia * 10 + id_algoritmo",
    }


def huella(parametros):
    texto = json.dumps(parametros, sort_keys=True)
    return hashlib.md5(texto.encode()).hexdigest()[:8]


def numero_de_instancia(instance_id):
    return int(instance_id.split("_")[-1])


def semilla(instance_id, algoritmo):
    return (config.SEMILLA_BASE * 1_000_000 + numero_de_instancia(instance_id) * 10
            + config.IDS_ALGORITMO[algoritmo])


def etiquetar_instancia(ruta_datos, instance_id):
    puntos = cargar_puntos(ruta_datos, instance_id)
    n = len(puntos)
    D = matriz_distancias(puntos)
    costo_hc, _ = hill_climbing_2opt(D, semilla(instance_id, "2opt_hill_climbing"))
    costo_mst, _, costo_mst_sin_mejora = mst_2opt(D)
    n_iter = iteraciones_sa(n, config.SA_FACTOR_ITER, config.SA_MIN_ITER)
    costo_sa, _ = recocido_simulado_2opt(D, config.SA_T0, config.SA_TF, n_iter,
                                         semilla(instance_id, "sa_2opt"))
    fila = {"instance_id": instance_id,
            "cost_2opt_hill_climbing": costo_hc,
            "cost_mst_2opt": costo_mst,
            "cost_sa_2opt": costo_sa,
            "cost_mst_sin_2opt": costo_mst_sin_mejora,
            "sa_iteraciones": n_iter}
    fila.update(caracteristicas_extra(puntos, D))
    return fila


def elegir_mejor(fila):
    costos = [fila[c] for c in COLUMNAS_COSTO]
    minimo = min(costos)
    empatados = [a for a, c in zip(config.ALGORITMOS, costos)
                 if c - minimo <= config.TOLERANCIA_EMPATE]
    return empatados[0], len(empatados)


def agregar_etiquetas(df):
    mejores = df.apply(lambda f: elegir_mejor(f), axis=1)
    df["best_algorithm"] = [m[0] for m in mejores]
    df["n_empatados"] = [m[1] for m in mejores]
    df["hay_empate"] = df["n_empatados"] > 1
    return df


def balance(df, titulo):
    cuentas = df["best_algorithm"].value_counts().reindex(config.ALGORITMOS, fill_value=0)
    porc = (cuentas / len(df) * 100).round(2)
    print(f"{titulo} (n = {len(df)}):")
    for a in config.ALGORITMOS:
        print(f"   {a:<20} {cuentas[a]:>4}  ({porc[a]:>5.1f}%)")
    return cuentas, porc


def main():
    ap = argparse.ArgumentParser(description="Parte 1: genera las etiquetas del dataset.")
    ap.add_argument("--datos", default=config.RUTA_DATOS, help="carpeta data/ del dataset")
    ap.add_argument("--salida", default=config.RUTA_SALIDA, help="carpeta de resultados")
    ap.add_argument("--limite", type=int, default=None,
                    help="solo las primeras N instancias de cada archivo (para pruebas rapidas)")
    ap.add_argument("--reiniciar", action="store_true", help="ignora el cache y recalcula todo")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    train = pd.read_csv(os.path.join(args.datos, "train.csv"))
    test = pd.read_csv(os.path.join(args.datos, "test.csv"))
    if args.limite:
        train, test = train.head(args.limite), test.head(args.limite)
    ids = list(train["instance_id"]) + list(test["instance_id"])

    parametros = parametros_experimento()
    sello = huella(parametros)
    ruta_cache = os.path.join(args.salida, f"costos_cache_{sello}.csv")
    if args.reiniciar and os.path.exists(ruta_cache):
        os.remove(ruta_cache)

    hechos = {}
    if os.path.exists(ruta_cache):
        cache = pd.read_csv(ruta_cache)
        hechos = {r["instance_id"]: r for r in cache.to_dict("records")}
        print(f"Cache encontrado: {len(hechos)} instancias ya calculadas ({ruta_cache}).")

    pendientes = [i for i in ids if i not in hechos]
    print(f"Instancias totales: {len(ids)} | por calcular: {len(pendientes)}")
    print(f"Parametros SA: T0 = {config.SA_T0}, Tf = {config.SA_TF}, "
          f"N = max({config.SA_MIN_ITER}, {config.SA_FACTOR_ITER} * n^2)")

    inicio = time.perf_counter()
    nuevo = not os.path.exists(ruta_cache)
    with open(ruta_cache, "a", newline="") as f:
        escritor = None
        for k, instance_id in enumerate(pendientes, start=1):
            fila = etiquetar_instancia(args.datos, instance_id)
            if escritor is None:
                escritor = csv.DictWriter(f, fieldnames=list(fila.keys()))
                if nuevo:
                    escritor.writeheader()
            escritor.writerow(fila)
            f.flush()
            hechos[instance_id] = fila
            if k % 25 == 0 or k == len(pendientes):
                t = time.perf_counter() - inicio
                restante = t / k * (len(pendientes) - k)
                print(f"   {k}/{len(pendientes)} instancias | {t:6.1f} s transcurridos | "
                      f"~{restante:6.1f} s restantes", flush=True)

    costos = pd.DataFrame([hechos[i] for i in ids]).drop_duplicates("instance_id")
    train_et = train.merge(costos, on="instance_id", how="left")
    test_et = test.merge(costos, on="instance_id", how="left")
    train_et = agregar_etiquetas(train_et)
    test_et = agregar_etiquetas(test_et)
    train_et.to_csv(os.path.join(args.salida, "train_etiquetado.csv"), index=False)
    test_et.to_csv(os.path.join(args.salida, "test_etiquetado.csv"), index=False)

    print("\nBALANCE DE CLASES (algoritmo con menor costo en cada instancia)")
    todo = pd.concat([train_et, test_et], ignore_index=True)
    filas_balance = []
    for titulo, df in [("Todas las instancias", todo), ("Entrenamiento", train_et), ("Prueba", test_et)]:
        cuentas, porc = balance(df, titulo)
        for a in config.ALGORITMOS:
            filas_balance.append({"conjunto": titulo, "algoritmo": a, "instancias": int(cuentas[a]),
                                  "porcentaje": float(porc[a])})
    pd.DataFrame(filas_balance).to_csv(os.path.join(args.salida, "balance_clases.csv"), index=False)
    empates = int(todo["hay_empate"].sum())
    print(f"\nInstancias con empate en el menor costo: {empates} ({empates / len(todo) * 100:.1f}%). "
          f"Se resuelven con la prioridad {config.ALGORITMOS}.")

    parametros["ejecucion"] = {
        "instancias_train": len(train_et), "instancias_test": len(test_et),
        "instancias_con_empate": empates, "python": platform.python_version(),
        "sello_parametros": sello,
        "iteraciones_sa_por_n": {int(n): iteraciones_sa(int(n), config.SA_FACTOR_ITER, config.SA_MIN_ITER)
                                 for n in sorted(todo["n"].unique())},
    }
    with open(os.path.join(args.salida, "parametros_utilizados.json"), "w") as f:
        json.dump(parametros, f, indent=2, ensure_ascii=False)
    print(f"\nArchivos escritos en '{args.salida}': train_etiquetado.csv, test_etiquetado.csv, "
          f"balance_clases.csv, parametros_utilizados.json")


if __name__ == "__main__":
    sys.exit(main())
