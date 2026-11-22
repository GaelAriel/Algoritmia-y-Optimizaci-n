import math
import random

EPS = 1e-12


def matriz_distancias(puntos):
    n = len(puntos)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = puntos[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - puntos[j][0], yi - puntos[j][1])
            D[i][j] = d
            D[j][i] = d
    return D


def longitud_tour(D, tour):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))


def busqueda_local_2opt(D, tour):
    """2-opt de primera mejora hasta llegar a un optimo local.
    Devuelve (tour, numero_de_mejoras). La posicion 0 queda fija."""
    tour = list(tour)
    n = len(tour)
    mejoras = 0
    hubo_mejora = True
    while hubo_mejora:
        hubo_mejora = False
        for i in range(1, n - 1):
            a = tour[i - 1]
            Da = D[a]
            b = tour[i]
            Dab = Da[b]
            for k in range(i + 1, n):
                c = tour[k]
                d = tour[(k + 1) % n]
                delta = Da[c] + D[b][d] - Dab - D[c][d]
                if delta < -EPS:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    mejoras += 1
                    hubo_mejora = True
                    b = tour[i]
                    Dab = Da[b]
    return tour, mejoras


def tour_aleatorio(n, rng):
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def hill_climbing_2opt(D, semilla):
    """Algoritmo 1: 2-opt (subida de colinas) desde un tour aleatorio."""
    rng = random.Random(semilla)
    n = len(D)
    tour, _ = busqueda_local_2opt(D, tour_aleatorio(n, rng))
    return longitud_tour(D, tour), tour


def arbol_expansion_minima(D):
    """Prim desde la ciudad 0. Devuelve {nodo: [hijos]}."""
    n = len(D)
    mejor = {v: (D[0][v], 0) for v in range(1, n)}
    hijos = {v: [] for v in range(n)}
    while mejor:
        v = min(mejor, key=lambda u: mejor[u][0])
        hijos[mejor.pop(v)[1]].append(v)
        for u in mejor:
            if D[v][u] < mejor[u][0]:
                mejor[u] = (D[v][u], v)
    return hijos


def preorden(hijos, raiz=0):
    resultado = []
    pila = [raiz]
    while pila:
        v = pila.pop()
        resultado.append(v)
        pila.extend(reversed(hijos[v]))
    return resultado


def mst_2opt(D):
    """Algoritmo 3: aproximacion por arbol de expansion minima + 2-opt."""
    tour_mst = preorden(arbol_expansion_minima(D))
    tour, _ = busqueda_local_2opt(D, tour_mst)
    return longitud_tour(D, tour), tour, longitud_tour(D, tour_mst)


def iteraciones_sa(n, factor, minimo):
    return max(minimo, int(factor * n * n))


def recocido_simulado_2opt(D, T0, Tf, n_iter, semilla):
    """Algoritmo 2: recocido simulado con movimiento 2-opt aleatorio, tour inicial
    aleatorio y enfriamiento geometrico de T0 a Tf en n_iter iteraciones.
    Devuelve (mejor_longitud, tour_final_actual)."""
    rng = random.Random(semilla)
    n = len(D)
    actual = tour_aleatorio(n, rng)
    L = longitud_tour(D, actual)
    mejor = L
    factor = (Tf / T0) ** (1.0 / n_iter)
    T = T0
    exp = math.exp
    muestra = rng.sample
    aleatorio = rng.random
    rango = range(1, n)
    for _ in range(n_iter):
        i, k = muestra(rango, 2)
        if i > k:
            i, k = k, i
        if i == 1 and k == n - 1:
            T *= factor
            continue
        a = actual[i - 1]
        b = actual[i]
        c = actual[k]
        d = actual[(k + 1) % n]
        dL = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if dL < 0 or aleatorio() < exp(-dL / T):
            actual[i:k + 1] = actual[i:k + 1][::-1]
            L += dL
            if L < mejor:
                mejor = L
        T *= factor
    return mejor, actual
