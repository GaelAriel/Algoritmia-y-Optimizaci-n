import os

RUTA_DATOS = os.environ.get("HH_DATA_DIR", "hh_tsp_dataset/data")
RUTA_SALIDA = os.environ.get("HH_OUT_DIR", "resultados")

SEMILLA_BASE = 42

ALGORITMOS = ["2opt_hill_climbing", "mst_2opt", "sa_2opt"]
COLUMNAS_COSTO = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "mst_2opt": "cost_mst_2opt",
    "sa_2opt": "cost_sa_2opt",
}
IDS_ALGORITMO = {"2opt_hill_climbing": 1, "mst_2opt": 2, "sa_2opt": 3}

# Dos costos que difieren menos que esta tolerancia se consideran empate.
# En empate gana el algoritmo que aparece primero en ALGORITMOS (el mas barato).
TOLERANCIA_EMPATE = 1e-9

# Recocido simulado (coordenadas en [0, 1]^2).
# T0 = 0.14 y Tf = 0.001 son los valores de la Tarea 3 (nivel "bajo" de T0 = 1.377 y
# Tf = 0.01 sobre coordenadas en [0, 10]^2) escalados por 1/10.
SA_T0 = float(os.environ.get("HH_SA_T0", 0.14))
SA_TF = float(os.environ.get("HH_SA_TF", 0.001))
SA_FACTOR_ITER = float(os.environ.get("HH_SA_FACTOR", 5))
SA_MIN_ITER = int(os.environ.get("HH_SA_MIN_ITER", 300))

# Machine Learning
SEMILLA_ML = 42
CV_FOLDS = 5
# Metrica con la que se eligen hiperparametros y modelo (solo con el conjunto de
# entrenamiento): "regret" | "regret_rel" | "accuracy" | "balanced_accuracy"
METRICA_SELECCION = "regret"
REPETICIONES_PERMUTACION = 30
REMUESTREOS_BOOTSTRAP = 2000

CARACTERISTICAS_BASE = ["n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist"]
CARACTERISTICAS_EXTRA = ["clark_evans", "cv_nn_dist", "mean_border_dist", "bbox_area"]
CONJUNTOS_CARACTERISTICAS = {
    "base": {"num": CARACTERISTICAS_BASE, "cat": ["distribution"]},
    "extendido": {"num": CARACTERISTICAS_BASE + CARACTERISTICAS_EXTRA, "cat": ["distribution"]},
}
MODELOS = ["arbol", "knn", "bosque"]
