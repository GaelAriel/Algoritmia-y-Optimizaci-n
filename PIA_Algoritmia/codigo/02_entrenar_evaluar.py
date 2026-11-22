import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import config

ALGS = config.ALGORITMOS
COLS = [config.COLUMNAS_COSTO[a] for a in ALGS]
TOL = config.TOLERANCIA_EMPATE
NOMBRES_MODELO = {"arbol": "Arbol de decision", "knn": "KNN", "bosque": "Bosque aleatorio"}
COLORES = {"2opt_hill_climbing": "#005A8C", "mst_2opt": "#E08E0B", "sa_2opt": "#27AE60"}


# ---------------------------------------------------------------- datos y metricas
def cargar_datos(carpeta):
    train = pd.read_csv(os.path.join(carpeta, "train_etiquetado.csv")).set_index("instance_id")
    test = pd.read_csv(os.path.join(carpeta, "test_etiquetado.csv")).set_index("instance_id")
    return train, test


def matriz_costos(df):
    return df[COLS]


def regret(costos, idx, predicciones):
    """regret = costo del algoritmo elegido - costo del mejor algoritmo real."""
    C = costos.loc[idx, COLS].to_numpy()
    elegido = C[np.arange(len(C)), [ALGS.index(p) for p in predicciones]]
    mejor = C.min(axis=1)
    return elegido - mejor, (elegido - mejor) / mejor


def crear_scorers(costos):
    def sc_regret(estimador, X, y=None):
        r, _ = regret(costos, X.index, estimador.predict(X))
        return -float(np.mean(r))

    def sc_regret_rel(estimador, X, y=None):
        _, r = regret(costos, X.index, estimador.predict(X))
        return -float(np.mean(r))

    return {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
            "regret": sc_regret, "regret_rel": sc_regret_rel}


def metricas(costos, idx, y_real, pred):
    r, r_rel = regret(costos, idx, pred)
    return {
        "accuracy": float(accuracy_score(y_real, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, pred)),
        "f1_macro": float(f1_score(y_real, pred, labels=ALGS, average="macro", zero_division=0)),
        "accuracy_con_empates": float(np.mean(r <= TOL)),
        "regret_medio": float(np.mean(r)),
        "regret_mediano": float(np.median(r)),
        "regret_p90": float(np.percentile(r, 90)),
        "regret_max": float(np.max(r)),
        "regret_relativo_medio_pct": float(np.mean(r_rel) * 100),
    }


# ---------------------------------------------------------------- modelos
def construir_pipeline(modelo, conjunto):
    cfg = config.CONJUNTOS_CARACTERISTICAS[conjunto]
    escalador = StandardScaler() if modelo == "knn" else "passthrough"
    pre = ColumnTransformer([("num", escalador, cfg["num"]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["cat"])])
    if modelo == "arbol":
        clf = DecisionTreeClassifier(random_state=config.SEMILLA_ML)
    elif modelo == "knn":
        clf = KNeighborsClassifier()
    else:
        clf = RandomForestClassifier(random_state=config.SEMILLA_ML, n_jobs=1)
    return Pipeline([("pre", pre), ("clf", clf)])


def cuadricula(modelo, rapido):
    if modelo == "arbol":
        if rapido:
            return {"clf__max_depth": [3, 6, None], "clf__min_samples_leaf": [1, 10],
                    "clf__class_weight": [None, "balanced"]}
        return {"clf__max_depth": [2, 3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10, 20],
                "clf__class_weight": [None, "balanced"], "clf__criterion": ["gini", "entropy"]}
    if modelo == "knn":
        if rapido:
            return {"clf__n_neighbors": [3, 9], "clf__weights": ["uniform", "distance"]}
        return {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21], "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2]}
    if rapido:
        return {"clf__n_estimators": [100], "clf__max_depth": [5, None], "clf__min_samples_leaf": [1, 5]}
    return {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 3, 5], "clf__class_weight": [None, "balanced_subsample"]}


def X_de(df, conjunto):
    cfg = config.CONJUNTOS_CARACTERISTICAS[conjunto]
    return df[cfg["num"] + cfg["cat"]]


def hiperparametros_legibles(params):
    return {k.replace("clf__", ""): v for k, v in params.items()}


def seleccionar(train, rapido):
    costos = matriz_costos(train)
    scorers = crear_scorers(costos)
    y = train["best_algorithm"]
    minimo = int(y.value_counts().min())
    folds = max(2, min(config.CV_FOLDS, minimo))
    if folds < config.CV_FOLDS:
        print(f"AVISO: la clase menos frecuente tiene {minimo} instancias; se usan {folds} particiones.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEMILLA_ML)
    candidatos = []
    for conjunto in config.CONJUNTOS_CARACTERISTICAS:
        for modelo in config.MODELOS:
            print(f"   buscando hiperparametros: {NOMBRES_MODELO[modelo]} / caracteristicas '{conjunto}'",
                  flush=True)
            gs = GridSearchCV(construir_pipeline(modelo, conjunto), cuadricula(modelo, rapido),
                              scoring=scorers, refit=config.METRICA_SELECCION, cv=cv, n_jobs=1)
            gs.fit(X_de(train, conjunto), y)
            i = gs.best_index_
            res = gs.cv_results_
            candidatos.append({
                "modelo": modelo, "conjunto": conjunto, "estimador": gs.best_estimator_,
                "hiperparametros": hiperparametros_legibles(gs.best_params_),
                "cv_accuracy": float(res["mean_test_accuracy"][i]),
                "cv_balanced_accuracy": float(res["mean_test_balanced_accuracy"][i]),
                "cv_regret": float(-res["mean_test_regret"][i]),
                "cv_regret_rel_pct": float(-res["mean_test_regret_rel"][i] * 100),
                "cv_regret_std": float(res["std_test_regret"][i]),
            })
    clave = {"regret": lambda c: c["cv_regret"], "regret_rel": lambda c: c["cv_regret_rel_pct"],
             "accuracy": lambda c: -c["cv_accuracy"], "balanced_accuracy": lambda c: -c["cv_balanced_accuracy"]}
    elegido = min(candidatos, key=clave[config.METRICA_SELECCION])
    return candidatos, elegido, folds


# ---------------------------------------------------------------- evaluacion
def estrategias_regret(costos_test, idx, y_test, mayoritaria, pred_modelo):
    filas = []
    estrategias = [("Modelo (hiper-heuristica)", pred_modelo),
                   (f"Baseline: siempre {mayoritaria}", [mayoritaria] * len(idx))]
    for a in ALGS:
        if a != mayoritaria:
            estrategias.append((f"Siempre {a}", [a] * len(idx)))
    for nombre, pred in estrategias:
        m = metricas(costos_test, idx, y_test, pred)
        filas.append({"estrategia": nombre, **m})
    mejor = costos_test.loc[idx, COLS].min(axis=1)
    filas.append({"estrategia": "Oraculo (mejor algoritmo real)", "accuracy": 1.0, "balanced_accuracy": 1.0,
                  "f1_macro": 1.0, "accuracy_con_empates": 1.0, "regret_medio": 0.0, "regret_mediano": 0.0,
                  "regret_p90": 0.0, "regret_max": 0.0, "regret_relativo_medio_pct": 0.0})
    return pd.DataFrame(filas)


def bootstrap_diferencias(costos_test, idx, y_test, pred_modelo, pred_base, rng):
    r_m, _ = regret(costos_test, idx, pred_modelo)
    r_b, _ = regret(costos_test, idx, pred_base)
    ok_m = (np.array(pred_modelo) == y_test.to_numpy()).astype(float)
    ok_b = (np.array(pred_base) == y_test.to_numpy()).astype(float)
    n = len(idx)
    d_regret, d_acc = [], []
    for _ in range(config.REMUESTREOS_BOOTSTRAP):
        s = rng.integers(0, n, n)
        d_regret.append(r_b[s].mean() - r_m[s].mean())
        d_acc.append(ok_m[s].mean() - ok_b[s].mean())
    f = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"reduccion_regret_medio": float(r_b.mean() - r_m.mean()), "reduccion_regret_ic95": f(d_regret),
            "mejora_accuracy": float(ok_m.mean() - ok_b.mean()), "mejora_accuracy_ic95": f(d_acc)}


def analisis_casos_no_hc(costos_test, idx, pred):
    C = costos_test.loc[idx, COLS]
    c_hc = C[config.COLUMNAS_COSTO["2opt_hill_climbing"]].to_numpy()
    mejor = C.min(axis=1).to_numpy()
    elegido = C.to_numpy()[np.arange(len(C)), [ALGS.index(p) for p in pred]]
    pred = np.array(pred)
    hay_mejor = (c_hc - mejor) > TOL
    res = {"instancias_donde_algo_supera_a_2opt": int(hay_mejor.sum()),
           "instancias_donde_2opt_es_de_los_mejores": int((~hay_mejor).sum())}
    if hay_mejor.any():
        res["de_ellas_el_modelo_elige_algo_que_supera_a_2opt"] = int(((c_hc - elegido) > TOL)[hay_mejor].sum())
        res["de_ellas_el_modelo_elige_2opt"] = int((pred[hay_mejor] == "2opt_hill_climbing").sum())
        potencial = float((c_hc - mejor)[hay_mejor].sum())
        captado = float((c_hc - elegido)[hay_mejor].sum())
        res["fraccion_de_la_mejora_potencial_captada"] = captado / potencial if potencial > 0 else None
    if (~hay_mejor).any():
        res["falsas_alarmas_(modelo_elige_algo_peor_que_2opt)"] = int(((elegido - c_hc) > TOL)[~hay_mejor].sum())
    return res


def importancia_agregada(pipeline):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    nombres = pipeline.named_steps["pre"].get_feature_names_out()
    cats = [c for cfg in config.CONJUNTOS_CARACTERISTICAS.values() for c in cfg["cat"]]
    agregada = {}
    for nombre, valor in zip(nombres, clf.feature_importances_):
        base = nombre.split("__", 1)[1]
        for c in set(cats):
            if base.startswith(c + "_"):
                base = c
        agregada[base] = agregada.get(base, 0.0) + float(valor)
    return pd.Series(agregada).sort_values(ascending=False)


# ---------------------------------------------------------------- figuras
def fig_balance(balance, ruta):
    conjuntos = ["Todas las instancias", "Entrenamiento", "Prueba"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ancho = 0.26
    for k, a in enumerate(ALGS):
        vals = [balance[(balance.conjunto == c) & (balance.algoritmo == a)]["porcentaje"].iloc[0] for c in conjuntos]
        barras = ax.bar([i + (k - 1) * ancho for i in range(3)], vals, ancho, color=COLORES[a], label=a)
        ax.bar_label(barras, fmt="%.1f%%", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(conjuntos)
    ax.set_ylabel("% de instancias donde gana")
    ax.set_title("Balance de clases (algoritmo con menor costo)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_confusion(cm, ruta, titulo):
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, datos, fmt, t in [(axes[0], cm, "d", "Conteos"), (axes[1], norm, ".2f", "Normalizada por fila")]:
        im = ax.imshow(datos, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(ALGS, rotation=20, fontsize=8)
        ax.set_yticklabels(ALGS, fontsize=8)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Etiqueta real")
        ax.set_title(t)
        for i in range(3):
            for j in range(3):
                v = datos[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > datos.max() / 2 else "black")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_regret(tabla, regrets, ruta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    nombres = list(tabla["estrategia"])
    corto = []
    for n in nombres:
        if n.startswith("Modelo"):
            corto.append("Modelo")
        elif n.startswith("Baseline"):
            corto.append("Baseline: " + n.split("siempre ")[1])
        elif n.startswith("Oraculo"):
            corto.append("Oraculo")
        else:
            corto.append(n)
    axes[0].barh(range(len(nombres)), tabla["regret_medio"], color="#005A8C")
    axes[0].set_yticks(range(len(nombres)))
    axes[0].set_yticklabels(corto, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Regret medio en prueba (menor es mejor)")
    axes[0].set_title("Regret medio por estrategia")
    axes[0].grid(axis="x", alpha=0.3)
    datos = [regrets[n] for n in regrets]
    axes[1].boxplot(datos, tick_labels=list(regrets.keys()), showmeans=True)
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_ylabel("Regret por instancia (escala symlog)")
    axes[1].set_title("Distribucion del regret: modelo vs baseline")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_importancia(perm, impureza, ruta):
    paneles = 3 if impureza is not None else 2
    fig, axes = plt.subplots(1, paneles, figsize=(5 * paneles, 4.2))
    p = perm.sort_values("aumento_regret", ascending=True)
    axes[0].barh(p.index, p["aumento_regret"], xerr=p["aumento_regret_desv"], color="#C0392B")
    axes[0].set_title("Permutacion: aumento del regret")
    p = perm.sort_values("caida_accuracy", ascending=True)
    axes[1].barh(p.index, p["caida_accuracy"], xerr=p["caida_accuracy_desv"], color="#005A8C")
    axes[1].set_title("Permutacion: caida del accuracy")
    if impureza is not None:
        s = impureza.sort_values(ascending=True)
        axes[2].barh(s.index, s.values, color="#27AE60")
        axes[2].set_title("Importancia por impureza (modelo)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Importancia de caracteristicas (evaluada en prueba)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_patrones(patron_pct, tamanos, ruta):
    dists = list(patron_pct.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, len(dists), figsize=(4.6 * len(dists), 5), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dists):
        M = patron_pct.loc[d].reindex(tamanos)[ALGS].to_numpy()
        im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["2opt", "mst", "sa"])
        ax.set_yticks(range(len(tamanos)))
        ax.set_yticklabels(tamanos)
        ax.set_title(d)
        ax.set_xlabel("Algoritmo ganador")
        for i in range(M.shape[0]):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        color="white" if M[i, j] > 55 else "black", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("n (numero de ciudades)")
    fig.suptitle("% de instancias que gana cada algoritmo por distribucion y tamano (train + test)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def fig_arbol(pipeline, ruta_png, ruta_txt, titulo):
    clf = pipeline.named_steps["clf"]
    nombres = list(pipeline.named_steps["pre"].get_feature_names_out())
    nombres = [n.split("__", 1)[1] for n in nombres]
    mostrar = min(clf.get_depth(), 4)
    ancho = min(20, max(8.0, 2.8 * 2 ** mostrar))
    fig, ax = plt.subplots(figsize=(ancho, 2.6 + 1.5 * mostrar))
    plot_tree(clf, feature_names=nombres, class_names=list(clf.classes_), filled=True, rounded=True,
              max_depth=4, fontsize=12 if mostrar <= 2 else 9, ax=ax)
    ax.set_title(titulo + " (se muestran hasta 4 niveles)")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)
    with open(ruta_txt, "w") as f:
        f.write(export_text(clf, feature_names=nombres))


# ---------------------------------------------------------------- tablas markdown
def md_tabla(df, decimales=3):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.{decimales}f}")
    cab = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    filas = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([cab, sep] + filas)


def main():
    ap = argparse.ArgumentParser(description="Parte 2: entrena y evalua el clasificador (hiper-heuristica).")
    ap.add_argument("--salida", default=config.RUTA_SALIDA, help="carpeta con train/test_etiquetado.csv")
    ap.add_argument("--rapido", action="store_true", help="cuadriculas reducidas (solo para probar)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    train, test = cargar_datos(args.salida)
    costos_train, costos_test = matriz_costos(train), matriz_costos(test)
    idx_test = test.index
    y_test = test["best_algorithm"]
    rng = np.random.default_rng(config.SEMILLA_ML)
    with open(os.path.join(args.salida, "parametros_utilizados.json")) as f:
        parametros = json.load(f)

    print("1) Balance de clases y baseline")
    balance = pd.read_csv(os.path.join(args.salida, "balance_clases.csv"))
    todo = pd.concat([train, test])
    mayoritaria = train["best_algorithm"].value_counts().idxmax()
    pred_base = [mayoritaria] * len(test)
    print(f"   clase mayoritaria en entrenamiento: {mayoritaria} -> baseline = siempre predecirla")
    fig_balance(balance, os.path.join(args.salida, "fig_balance_clases.png"))

    print("2) Seleccion de modelo e hiperparametros (validacion cruzada solo con entrenamiento)")
    candidatos, elegido, folds = seleccionar(train, args.rapido)
    sel = pd.DataFrame([{k: v for k, v in c.items() if k not in ("estimador",)} for c in candidatos])
    sel["hiperparametros"] = sel["hiperparametros"].map(lambda d: json.dumps(d, ensure_ascii=False))
    sel["seleccionado"] = [c is elegido for c in candidatos]
    sel.to_csv(os.path.join(args.salida, "seleccion_modelo_cv.csv"), index=False)
    print(f"   modelo elegido: {NOMBRES_MODELO[elegido['modelo']]} / caracteristicas '{elegido['conjunto']}' "
          f"(criterio: {config.METRICA_SELECCION}, {folds} particiones)")
    print(f"   hiperparametros: {elegido['hiperparametros']}")

    print("3) Evaluacion en el conjunto de prueba")
    modelo = elegido["estimador"]
    X_test = X_de(test, elegido["conjunto"])
    pred = list(modelo.predict(X_test))
    cm = confusion_matrix(y_test, pred, labels=ALGS)
    cm_base = confusion_matrix(y_test, pred_base, labels=ALGS)
    pd.DataFrame(cm, index=[f"real_{a}" for a in ALGS], columns=[f"pred_{a}" for a in ALGS]).to_csv(
        os.path.join(args.salida, "matriz_confusion.csv"))
    fig_confusion(cm, os.path.join(args.salida, "fig_matriz_confusion.png"),
                  f"Matriz de confusion en prueba: {NOMBRES_MODELO[elegido['modelo']]}")
    tabla = estrategias_regret(costos_test, idx_test, y_test, mayoritaria, pred)
    tabla.to_csv(os.path.join(args.salida, "regret_y_metricas_por_estrategia.csv"), index=False)
    r_modelo, _ = regret(costos_test, idx_test, pred)
    r_base, _ = regret(costos_test, idx_test, pred_base)
    fig_regret(tabla, {"Modelo": r_modelo, "Baseline": r_base}, os.path.join(args.salida, "fig_regret.png"))
    comp = bootstrap_diferencias(costos_test, idx_test, y_test, pred, pred_base, rng)
    no_hc = analisis_casos_no_hc(costos_test, idx_test, pred)
    dist_pred = pd.Series(pred).value_counts().reindex(ALGS, fill_value=0)
    colapsa = bool(dist_pred.max() / len(pred) >= 0.95)

    print("4) Importancia de caracteristicas")
    cfg = config.CONJUNTOS_CARACTERISTICAS[elegido["conjunto"]]
    sc = crear_scorers(costos_test)
    res = permutation_importance(modelo, X_test, y_test, scoring={"regret": sc["regret"], "accuracy": "accuracy"},
                                 n_repeats=config.REPETICIONES_PERMUTACION, random_state=config.SEMILLA_ML)
    perm = pd.DataFrame({"aumento_regret": res["regret"].importances_mean,
                         "aumento_regret_desv": res["regret"].importances_std,
                         "caida_accuracy": res["accuracy"].importances_mean,
                         "caida_accuracy_desv": res["accuracy"].importances_std}, index=cfg["num"] + cfg["cat"])
    perm = perm.sort_values("aumento_regret", ascending=False)
    perm.to_csv(os.path.join(args.salida, "importancia_permutacion.csv"))
    impureza = importancia_agregada(modelo)
    if impureza is not None:
        impureza.to_csv(os.path.join(args.salida, "importancia_impureza.csv"), header=["importancia"])
    fig_importancia(perm, impureza, os.path.join(args.salida, "fig_importancia_caracteristicas.png"))

    print("5) Patrones por distribucion y tamano")
    tamanos = sorted(todo["n"].unique())
    patron = (todo.groupby(["distribution", "n", "best_algorithm"]).size().unstack(fill_value=0)
              .reindex(columns=ALGS, fill_value=0))
    total_grupo = patron.sum(axis=1)
    patron_pct = patron.div(total_grupo, axis=0) * 100
    tabla_patron = patron_pct.copy()
    tabla_patron.columns = [f"pct_{a}" for a in ALGS]
    tabla_patron["instancias"] = total_grupo
    tabla_patron["pct_con_empate"] = todo.groupby(["distribution", "n"])["hay_empate"].mean() * 100
    tabla_patron.round(2).to_csv(os.path.join(args.salida, "patrones_distribucion_tamano.csv"))
    fig_patrones(patron_pct, tamanos, os.path.join(args.salida, "fig_patrones_ganador.png"))
    por_dist = todo.groupby(["distribution", "best_algorithm"]).size().unstack(fill_value=0).reindex(
        columns=ALGS, fill_value=0)
    por_dist = por_dist.div(por_dist.sum(axis=1), axis=0) * 100
    por_n = todo.groupby(["n", "best_algorithm"]).size().unstack(fill_value=0).reindex(columns=ALGS, fill_value=0)
    por_n = por_n.div(por_n.sum(axis=1), axis=0) * 100
    brechas = {}
    for a in ALGS:
        col = config.COLUMNAS_COSTO[a]
        brechas[a] = (todo[col] / todo[COLS].min(axis=1) - 1) * 100
    brecha_tabla = pd.DataFrame(brechas).groupby([todo["distribution"], todo["n"]]).mean().round(2)
    brecha_tabla.to_csv(os.path.join(args.salida, "brecha_media_por_distribucion_tamano.csv"))
    por_grupo_test = []
    for clave, nombre in [("distribution", "distribucion"), ("n", "n")]:
        for valor, sub in test.groupby(clave):
            ids = sub.index
            p = pd.Series(pred, index=idx_test).loc[ids]
            rm, _ = regret(costos_test, ids, p)
            rb, _ = regret(costos_test, ids, [mayoritaria] * len(ids))
            por_grupo_test.append({"agrupacion": nombre, "valor": valor, "instancias_prueba": len(ids),
                                   "accuracy_modelo": float(accuracy_score(sub["best_algorithm"], p)),
                                   "accuracy_baseline": float(accuracy_score(sub["best_algorithm"],
                                                                             [mayoritaria] * len(ids))),
                                   "regret_medio_modelo": float(rm.mean()),
                                   "regret_medio_baseline": float(rb.mean())})
    pd.DataFrame(por_grupo_test).to_csv(os.path.join(args.salida, "desempeno_prueba_por_grupo.csv"), index=False)

    print("6) Comparacion de todos los candidatos en prueba (solo informativa; la eleccion fue por validacion cruzada)")
    filas = []
    for c in candidatos:
        pr = list(c["estimador"].predict(X_de(test, c["conjunto"])))
        m = metricas(costos_test, idx_test, y_test, pr)
        filas.append({"modelo": c["modelo"], "conjunto": c["conjunto"], "seleccionado_por_cv": c is elegido,
                      "cv_regret": c["cv_regret"], "test_accuracy": m["accuracy"],
                      "test_regret_medio": m["regret_medio"], "test_balanced_accuracy": m["balanced_accuracy"]})
    pd.DataFrame(filas).to_csv(os.path.join(args.salida, "comparacion_modelos_test.csv"), index=False)
    arbol_cand = [c for c in candidatos if c["modelo"] == "arbol"]
    mejor_arbol = min(arbol_cand, key=lambda c: c["cv_regret"])
    fig_arbol(mejor_arbol["estimador"], os.path.join(args.salida, "fig_arbol_decision.png"),
              os.path.join(args.salida, "arbol_reglas.txt"),
              f"Arbol de decision (caracteristicas '{mejor_arbol['conjunto']}')")

    resumen = {
        "modelo_elegido": elegido["modelo"], "conjunto_caracteristicas": elegido["conjunto"],
        "hiperparametros": elegido["hiperparametros"], "metrica_de_seleccion": config.METRICA_SELECCION,
        "particiones_cv": folds, "clase_mayoritaria_entrenamiento": mayoritaria,
        "metricas_modelo_en_prueba": metricas(costos_test, idx_test, y_test, pred),
        "metricas_baseline_en_prueba": metricas(costos_test, idx_test, y_test, pred_base),
        "comparacion_bootstrap_modelo_vs_baseline": comp,
        "analisis_casos_donde_algo_supera_a_2opt": no_hc,
        "distribucion_de_predicciones": {a: int(dist_pred[a]) for a in ALGS},
        "el_modelo_colapsa_a_una_clase": colapsa,
        "matriz_confusion": cm.tolist(), "matriz_confusion_baseline": cm_base.tolist(),
        "orden_de_clases": ALGS,
    }
    with open(os.path.join(args.salida, "metricas.json"), "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    escribir_resumen(args.salida, parametros, balance, mayoritaria, sel, elegido, resumen, tabla, perm,
                     impureza, por_dist, por_n, tabla_patron, pd.DataFrame(por_grupo_test), cm)
    print(f"\nListo. Resultados en '{args.salida}' (empieza por resumen_resultados.md).")


def escribir_resumen(salida, parametros, balance, mayoritaria, sel, elegido, resumen, tabla, perm, impureza,
                     por_dist, por_n, tabla_patron, por_grupo, cm):
    m, b = resumen["metricas_modelo_en_prueba"], resumen["metricas_baseline_en_prueba"]
    comp, no_hc = resumen["comparacion_bootstrap_modelo_vs_baseline"], resumen["analisis_casos_donde_algo_supera_a_2opt"]
    sa = parametros["sa_2opt"]
    L = []
    L.append("# Resumen de resultados (generado automaticamente)\n")
    L.append("## 1. Parametros usados para generar las etiquetas\n")
    L.append(f"- 2-opt hill climbing: tour inicial {parametros['2opt_hill_climbing']['tour_inicial']}, "
             f"{parametros['2opt_hill_climbing']['variante']}.")
    L.append(f"- MST + 2-opt: {parametros['mst_2opt']['tour_inicial']}; {parametros['mst_2opt']['mejora']}.")
    L.append(f"- SA + 2-opt: T0 = {sa['T0']}, Tf = {sa['Tf']}, enfriamiento {sa['enfriamiento']}, "
             f"iteraciones {sa['iteraciones']}, tour inicial {sa['tour_inicial']}.")
    L.append(f"- Iteraciones de SA por tamano: {parametros['ejecucion']['iteraciones_sa_por_n']}")
    L.append(f"- Semilla base {parametros['semilla_base']}; {parametros['esquema_de_semillas']}.")
    L.append(f"- Empates (diferencia <= {parametros['regla_de_empate']['tolerancia']}): gana el primero de "
             f"{parametros['regla_de_empate']['prioridad']}. Instancias con empate: "
             f"{parametros['ejecucion']['instancias_con_empate']}.\n")
    L.append("## 2. Balance de clases (algoritmo con menor costo)\n")
    pivote = balance.pivot(index="algoritmo", columns="conjunto", values="porcentaje").reindex(ALGS)
    inst = balance.pivot(index="algoritmo", columns="conjunto", values="instancias").reindex(ALGS)
    t = pd.DataFrame({"algoritmo": ALGS})
    for c in ["Todas las instancias", "Entrenamiento", "Prueba"]:
        t[c] = [f"{int(inst.loc[a, c])} ({pivote.loc[a, c]:.1f}%)" for a in ALGS]
    L.append(md_tabla(t))
    L.append(f"\nBaseline trivial: siempre predecir **{mayoritaria}** (la clase mas frecuente en entrenamiento). "
             f"Accuracy del baseline en prueba: {b['accuracy']:.3f}.\n")
    L.append("## 3. Seleccion del modelo (validacion cruzada, solo entrenamiento)\n")
    L.append(f"Criterio de seleccion: `{resumen['metrica_de_seleccion']}` con {resumen['particiones_cv']} particiones.\n")
    cols = ["modelo", "conjunto", "cv_regret", "cv_regret_rel_pct", "cv_accuracy", "cv_balanced_accuracy", "seleccionado"]
    L.append(md_tabla(sel[cols]))
    L.append(f"\n**Modelo elegido:** {NOMBRES_MODELO[elegido['modelo']]} con caracteristicas "
             f"'{elegido['conjunto']}'. **Hiperparametros:** `{json.dumps(elegido['hiperparametros'])}`.\n")
    L.append("## 4. Resultados en el conjunto de prueba\n")
    L.append(f"- Accuracy del modelo: **{m['accuracy']:.3f}** vs baseline **{b['accuracy']:.3f}** "
             f"(mejora {comp['mejora_accuracy']:+.3f}, IC95% bootstrap "
             f"[{comp['mejora_accuracy_ic95'][0]:+.3f}, {comp['mejora_accuracy_ic95'][1]:+.3f}]).")
    L.append(f"- Accuracy balanceada: modelo {m['balanced_accuracy']:.3f} vs baseline {b['balanced_accuracy']:.3f}.")
    L.append(f"- Accuracy contando empates como aciertos (el algoritmo elegido tiene el costo minimo): "
             f"modelo {m['accuracy_con_empates']:.3f} vs baseline {b['accuracy_con_empates']:.3f}.")
    L.append(f"- Regret medio: modelo **{m['regret_medio']:.4f}** vs baseline **{b['regret_medio']:.4f}** "
             f"(reduccion {comp['reduccion_regret_medio']:.4f}, IC95% bootstrap "
             f"[{comp['reduccion_regret_ic95'][0]:.4f}, {comp['reduccion_regret_ic95'][1]:.4f}]).")
    L.append(f"- Regret relativo medio: modelo {m['regret_relativo_medio_pct']:.2f}% vs baseline "
             f"{b['regret_relativo_medio_pct']:.2f}%.\n")
    L.append("Matriz de confusion (filas = real, columnas = prediccion):\n")
    cmdf = pd.DataFrame(cm, columns=[f"pred_{a}" for a in ALGS])
    cmdf.insert(0, "real", ALGS)
    L.append(md_tabla(cmdf))
    L.append("\nRegret y metricas por estrategia:\n")
    L.append(md_tabla(tabla[["estrategia", "accuracy", "accuracy_con_empates", "regret_medio", "regret_mediano",
                             "regret_p90", "regret_max", "regret_relativo_medio_pct"]], 4))
    L.append(f"\nDistribucion de las predicciones del modelo: {resumen['distribucion_de_predicciones']}. "
             f"Colapsa a una sola clase (>= 95%): **{'SI' if resumen['el_modelo_colapsa_a_una_clase'] else 'NO'}**.\n")
    L.append("### Casos donde SA o MST+2opt superan a 2-opt\n")
    for k, v in no_hc.items():
        L.append(f"- {k.replace('_', ' ')}: {v if not isinstance(v, float) else round(v, 3)}")
    L.append("\n## 5. Importancia de caracteristicas (permutacion en prueba)\n")
    L.append(md_tabla(perm.reset_index().rename(columns={"index": "caracteristica"}), 4))
    if impureza is not None:
        L.append("\nImportancia por impureza (modelo elegido):\n")
        L.append(md_tabla(impureza.reset_index().rename(columns={"index": "caracteristica", 0: "importancia"}), 4))
    L.append("\n## 6. Patrones por distribucion y tamano (train + test)\n")
    L.append("Porcentaje de instancias que gana cada algoritmo por distribucion:\n")
    L.append(md_tabla(por_dist.reset_index(), 1))
    L.append("\nPor tamano n:\n")
    L.append(md_tabla(por_n.reset_index(), 1))
    L.append("\nDetalle por (distribucion, n): ver `patrones_distribucion_tamano.csv` y `fig_patrones_ganador.png`.\n")
    L.append("Desempeno en prueba por grupo:\n")
    L.append(md_tabla(por_grupo, 3))
    with open(os.path.join(salida, "resumen_resultados.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
