import csv
import itertools
import math
import os
import random
import sys

import pandas as pd

import config
from caracteristicas import caracteristicas_base, caracteristicas_extra
from tsp_algoritmos import (EPS, busqueda_local_2opt, hill_climbing_2opt, iteraciones_sa, longitud_tour,
                            matriz_distancias, mst_2opt, recocido_simulado_2opt, tour_aleatorio)


def cargar(ruta_datos, instance_id):
    with open(os.path.join(ruta_datos, "instances", f"{instance_id}.csv"), newline="") as f:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]


def optimo_fuerza_bruta(D):
    n = len(D)
    return min(longitud_tour(D, (0,) + p) for p in itertools.permutations(range(1, n)))


def es_optimo_local_2opt(D, tour):
    n = len(tour)
    base = longitud_tour(D, tour)
    for i in range(1, n - 1):
        for k in range(i + 1, n):
            nuevo = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
            if longitud_tour(D, nuevo) < base - 1e-9:
                return False
    return True


def main():
    ruta = config.RUTA_DATOS
    train = pd.read_csv(os.path.join(ruta, "train.csv"))
    errores = 0

    def verificar(condicion, mensaje):
        nonlocal errores
        estado = "OK " if condicion else "FALLA"
        if not condicion:
            errores += 1
        print(f"[{estado}] {mensaje}")

    print("1. Las caracteristicas recalculadas coinciden con las de train.csv")
    muestra = train.sample(12, random_state=1)
    maxdif = 0.0
    for _, fila in muestra.iterrows():
        P = cargar(ruta, fila["instance_id"])
        D = matriz_distancias(P)
        c = caracteristicas_base(P, D)
        for k, v in c.items():
            maxdif = max(maxdif, abs(v - fila[k]))
    verificar(maxdif < 1e-6, f"diferencia maxima en las 4 caracteristicas = {maxdif:.2e}")

    print("2. Validez de los tours y optimalidad local de 2-opt")
    for n in (5, 10, 20, 40):
        iid = train[train.n == n]["instance_id"].iloc[0]
        D = matriz_distancias(cargar(ruta, iid))
        c1, t1 = hill_climbing_2opt(D, 7)
        c3, t3, c_mst = mst_2opt(D)
        verificar(sorted(t1) == list(range(n)) and sorted(t3) == list(range(n)), f"n={n}: los tours son permutaciones validas")
        verificar(abs(longitud_tour(D, t1) - c1) < 1e-9, f"n={n}: el costo de 2-opt coincide con la longitud de su tour")
        verificar(es_optimo_local_2opt(D, t1) and es_optimo_local_2opt(D, t3),
                  f"n={n}: HC y MST+2-opt terminan en un optimo local 2-opt")
        verificar(c3 <= c_mst + 1e-12, f"n={n}: 2-opt no empeora el tour del MST ({c_mst:.4f} -> {c3:.4f})")

    print("3. Ningun algoritmo supera al optimo exacto (n <= 9), y SA lo alcanza con presupuesto suficiente")
    ok_cota, ok_sa = True, True
    for iid in train[train.n.isin([5])]["instance_id"].head(3).tolist() + \
            [train[train.n == 10]["instance_id"].iloc[0]]:
        P = cargar(ruta, iid)
        if len(P) > 9:
            P = P[:9]
        D = matriz_distancias(P)
        opt = optimo_fuerza_bruta(D)
        costos = [hill_climbing_2opt(D, 3)[0], mst_2opt(D)[0], recocido_simulado_2opt(D, 0.14, 0.001, 5000, 3)[0]]
        ok_cota &= all(c >= opt - 1e-9 for c in costos)
        ok_sa &= abs(costos[2] - opt) < 1e-9
    verificar(ok_cota, "los costos de los tres algoritmos son >= optimo exacto")
    verificar(ok_sa, "SA con 5000 iteraciones alcanza el optimo exacto en instancias de 5 a 9 ciudades")

    print("4. Determinismo: misma semilla, mismo resultado")
    D = matriz_distancias(cargar(ruta, train[train.n == 30]["instance_id"].iloc[0]))
    verificar(hill_climbing_2opt(D, 5)[0] == hill_climbing_2opt(D, 5)[0], "2-opt con la misma semilla")
    verificar(recocido_simulado_2opt(D, 0.14, 0.001, 3000, 5)[0] == recocido_simulado_2opt(D, 0.14, 0.001, 3000, 5)[0],
              "SA con la misma semilla")
    verificar(recocido_simulado_2opt(D, 0.14, 0.001, 3000, 5)[0] != recocido_simulado_2opt(D, 0.14, 0.001, 3000, 6)[0],
              "SA con otra semilla da otro resultado")

    print("5. Presupuesto de iteraciones del SA")
    verificar(iteraciones_sa(5, config.SA_FACTOR_ITER, config.SA_MIN_ITER) >= config.SA_MIN_ITER, "n=5 respeta el minimo")
    verificar(iteraciones_sa(100, config.SA_FACTOR_ITER, config.SA_MIN_ITER) == int(config.SA_FACTOR_ITER * 10000),
              "n=100 usa factor * n^2")

    print("6. Las caracteristicas extra estan en rangos validos")
    P = cargar(ruta, train[train.n == 40]["instance_id"].iloc[0])
    e = caracteristicas_extra(P, matriz_distancias(P))
    verificar(0 < e["bbox_area"] <= 1 and 0 <= e["mean_border_dist"] <= 0.5 and e["clark_evans"] > 0, f"{e}")

    print(f"\n{'TODAS LAS PRUEBAS PASARON' if errores == 0 else str(errores) + ' PRUEBA(S) FALLARON'}")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
