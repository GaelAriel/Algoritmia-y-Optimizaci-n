import math
import statistics


def _desv_poblacional(valores):
    if len(valores) < 2:
        return 0.0
    m = sum(valores) / len(valores)
    return math.sqrt(sum((v - m) ** 2 for v in valores) / len(valores))


def distancias_vecino_mas_cercano(D):
    n = len(D)
    return [min(D[i][j] for j in range(n) if j != i) for i in range(n)]


def caracteristicas_base(puntos, D):
    """Reproduce las 4 caracteristicas numericas que trae train.csv / test.csv."""
    n = len(puntos)
    pares = [D[i][j] for i in range(n) for j in range(i + 1, n)]
    cx = sum(p[0] for p in puntos) / n
    cy = sum(p[1] for p in puntos) / n
    al_centroide = [math.hypot(p[0] - cx, p[1] - cy) for p in puntos]
    nn = distancias_vecino_mas_cercano(D)
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    return {
        "mean_pairwise_dist": sum(pares) / len(pares),
        "std_coords": (_desv_poblacional(xs) + _desv_poblacional(ys)) / 2,
        "mean_centroid_dist": sum(al_centroide) / n,
        "mean_nn_dist": sum(nn) / n,
    }


def caracteristicas_extra(puntos, D):
    """Caracteristicas propias calculadas solo a partir de las coordenadas.
    clark_evans: distancia media al vecino mas cercano / la esperada con puntos
        uniformes en el cuadrado unitario (<1 agrupado, ~1 uniforme, >1 regular).
    cv_nn_dist: coeficiente de variacion de las distancias al vecino mas cercano.
    mean_border_dist: distancia media de los puntos al borde del cuadrado.
    bbox_area: area del rectangulo que contiene a todos los puntos."""
    n = len(puntos)
    nn = distancias_vecino_mas_cercano(D)
    media_nn = sum(nn) / n
    esperado = 0.5 / math.sqrt(n)
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    borde = [min(x, 1 - x, y, 1 - y) for x, y in puntos]
    return {
        "clark_evans": media_nn / esperado,
        "cv_nn_dist": (statistics.pstdev(nn) / media_nn) if media_nn > 0 else 0.0,
        "mean_border_dist": sum(borde) / n,
        "bbox_area": (max(xs) - min(xs)) * (max(ys) - min(ys)),
    }
