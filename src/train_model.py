"""Fase 3: comparación de modelos y optimización con Optuna, con registro de experimentos.

Para cada modelo se corre un estudio de Optuna (sampler TPE con semilla fija + MedianPruner):
  - cada trial se evalúa con TimeSeriesSplit SOBRE TRAIN (etapa optuna_cv);
  - después de cada fold se reporta el MAE medio acumulado y el pruner corta los trials que
    van peor que la mediana de los anteriores en ese mismo fold (estado = podado);
  - la mejor configuración se reentrena en todo train y recién ahí se mide en VALIDATION
    (etapa optuna_val), que queda afuera de la búsqueda y sirve para comparar modelos.
El test NO se carga acá (incluir_test=False por defecto): se toca una sola vez,
al final, con el modelo ya elegido.

Uso (desde la raíz del proyecto):
    python src/train_model.py                                  # todos los modelos, 50 trials c/u
    python src/train_model.py --n-trials 100
    python src/train_model.py --modelos ridge,xgboost,mlp --n-splits 5
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from preprocess import ROOT, RUTA_FINAL, TARGET, huella

SEMILLA = 42
RUTA_LOGS = ROOT / "results" / "experiment_logs.csv"

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Espacio de búsqueda (se documenta acá y queda registrado en el log)
# =============================================================================
# Los features de dataset_final.csv ya vienen transformados y escalados, así que
# Ridge/Lasso pueden usarse directo. Los árboles son invariantes al escalado.

def _mlp(**params) -> TransformedTargetRegressor:
    """Red neuronal (MLP) con el target estandarizado.

    El target está en kW (~10000) y los pesos de la red arrancan cerca de 0: sin estandarizarlo
    la red entrena muy mal. El escalado de y se ajusta solo con los datos de entrenamiento que
    recibe .fit() y viaja dentro del objeto, así que .predict() devuelve kW directamente."""
    return TransformedTargetRegressor(regressor=MLPRegressor(**params), transformer=StandardScaler())


# Optuna no acepta tuplas como opciones categóricas: se elige un nombre y se traduce.
_CAPAS_MLP = {"32": (32,), "64": (64,), "64-32": (64, 32), "128-64": (128, 64)}


def _espacio_ridge(t: optuna.Trial) -> dict:
    return {"alpha": t.suggest_float("alpha", 1e-2, 1e3, log=True)}


def _espacio_lasso(t: optuna.Trial) -> dict:
    return {"alpha": t.suggest_float("alpha", 1e-1, 1e3, log=True)}


def _espacio_rf(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": t.suggest_categorical("max_depth", [None, 4, 8, 12]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "max_features": t.suggest_float("max_features", 0.3, 1.0),
    }


def _espacio_xgb(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 5),
    }


def _espacio_mlp(t: optuna.Trial) -> dict:
    return {
        "hidden_layer_sizes": _CAPAS_MLP[t.suggest_categorical("hidden_layer_sizes", list(_CAPAS_MLP))],
        "activation": t.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),   # regularización L2
        "learning_rate_init": t.suggest_float("learning_rate_init", 1e-3, 1e-2, log=True),
        "batch_size": t.suggest_categorical("batch_size", [32, 64, 128]),
    }


ESPACIOS: dict[str, dict] = {
    "ridge": {"clase": Ridge, "fijos": {"random_state": SEMILLA}, "espacio": _espacio_ridge},
    "lasso": {"clase": Lasso, "fijos": {"random_state": SEMILLA, "max_iter": 100_000},
              "espacio": _espacio_lasso},
    "random_forest": {"clase": RandomForestRegressor, "fijos": {"random_state": SEMILLA, "n_jobs": -1},
                      "espacio": _espacio_rf},
    "xgboost": {"clase": XGBRegressor,
                "fijos": {"random_state": SEMILLA, "n_jobs": -1, "tree_method": "hist"},
                "espacio": _espacio_xgb},
    "mlp": {
        "clase": _mlp,
        # Sin early_stopping: scikit-learn lo haría con un 10 % ALEATORIO de train, que en una
        # serie temporal mezcla pasado y futuro. Corta por n_iter_no_change sobre la pérdida de train.
        "fijos": {"random_state": SEMILLA, "solver": "adam", "max_iter": 1000,
                  "n_iter_no_change": 20, "early_stopping": False},
        "espacio": _espacio_mlp,
    },
}


# =============================================================================
# Datos
# =============================================================================

def cargar_datos(ruta: str | Path = RUTA_FINAL, incluir_test: bool = False) -> pd.DataFrame:
    """Lee dataset_final.csv indexado por fecha. Por defecto descarta las filas de test:
    ni siquiera quedan en memoria. Pasar incluir_test=True solo para la evaluación final."""
    df = pd.read_csv(ruta, parse_dates=["fecha"]).sort_values("fecha").set_index("fecha")
    if not incluir_test:
        df = df[df["split"] != "test"]
    return df


_CARACTERES_PROHIBIDOS = re.compile(r'[\[\]{}<>",:]')


def nombres_seguros(columnas) -> list[str]:
    """Nombres de columna aceptados por todos los modelos.

    XGBoost rechaza  [ ] <  en los nombres de features (p. ej. 'tot l3, l4 y planta de co2');
    se reemplazan esos y otros caracteres problemáticos por '_' y, si dos nombres quedan
    iguales, se desambiguan con un sufijo. Es determinista: predict.py debe aplicar esta misma
    función a las columnas del preprocesamiento para que coincidan con las del entrenamiento."""
    vistos: dict[str, int] = {}
    salida = []
    for c in columnas:
        n = _CARACTERES_PROHIBIDOS.sub("_", str(c))
        k = vistos.get(n, 0)
        vistos[n] = k + 1
        salida.append(n if k == 0 else f"{n}__{k}")
    return salida


def particion(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) de un split. Features = todo menos el target y la columna `split`.
    Los nombres de columna pasan por nombres_seguros()."""
    d = df[df["split"] == split]
    X = d.drop(columns=[TARGET, "split"])
    X.columns = nombres_seguros(X.columns)
    return X, d[TARGET]


def _metricas(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    ok = ~np.isnan(pred)  # los primeros días de un rolling quedan en NaN
    y, pred = y[ok], pred[ok]
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


# =============================================================================
# Baselines
# =============================================================================

def predicciones_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Predicción de cada baseline para la fila D (que apunta al consumo de D+1).

    El target de la fila D es frio(D+1), así que frio(D) = target de la fila D-1:
      persistencia   : frio(D)                      = y.shift(1)
      media móvil 7d : promedio de frio(D-6 .. D)   = y.shift(1).rolling(7).mean()
    Solo usa información disponible en D: no hay fuga hacia el futuro."""
    if not (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Hay días faltantes en dataset_final.csv: shift(1) ya no equivale a 'ayer'.")
    y = df[TARGET]
    return pd.DataFrame({
        "baseline_persistencia": y.shift(1),
        "baseline_media_movil_7d": y.shift(1).rolling(7).mean(),
    })


# =============================================================================
# Registro de experimentos
# =============================================================================

COLUMNAS_LOG = [
    "corrida", "timestamp", "modelo",
    "etapa",             # baseline | optuna_cv | optuna_val
    "estado",            # completo | podado (trial de Optuna cortado por el pruner)
    "hiperparametros",
    "mae_cv", "std_cv", "n_folds",             # solo optuna_cv (TimeSeriesSplit sobre train)
    "mae_val", "rmse_val", "r2_val", "mae_train",
    "seg_entrenamiento", "n_features", "n_train", "n_val", "semilla", "dataset_sha256",
]


def registrar_experimento(fila: dict, ruta: str | Path = RUTA_LOGS) -> None:
    """Agrega una fila a results/experiment_logs.csv (lo crea si no existe).
    Si el log viene de una versión anterior del script (menos columnas), lo completa
    con las columnas nuevas sin perder las filas viejas."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nueva = pd.DataFrame([{c: fila.get(c, np.nan) for c in COLUMNAS_LOG}])
    if not ruta.exists():
        nueva.to_csv(ruta, index=False)
        return
    previo = pd.read_csv(ruta)
    if list(previo.columns) == COLUMNAS_LOG:
        nueva.to_csv(ruta, mode="a", header=False, index=False)
    else:
        extra = [c for c in previo.columns if c not in COLUMNAS_LOG]
        pd.concat([previo, nueva], ignore_index=True).reindex(columns=COLUMNAS_LOG + extra).to_csv(ruta, index=False)


# =============================================================================
# Optimización con Optuna
# =============================================================================

def _ajustar(clase: Callable, params: dict, X, y):
    with warnings.catch_warnings():
        # Lasso/MLP avisan "no convergió" en muchas configuraciones; el MAE decide igual.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return clase(**params).fit(X, y)


def _objetivo(esp: dict, X_tr: pd.DataFrame, y_tr: pd.Series, cv: TimeSeriesSplit,
              anotar: Callable) -> Callable[[optuna.Trial], float]:
    """Función objetivo: MAE medio sobre los folds de TimeSeriesSplit (solo train).
    Después de cada fold reporta el MAE medio acumulado; si el pruner decide que el trial
    no promete, se registra como 'podado' y se corta."""
    def objetivo(trial: optuna.Trial) -> float:
        params = {**esp["fijos"], **esp["espacio"](trial)}
        maes: list[float] = []
        t0 = time.perf_counter()
        for paso, (i_fit, i_eval) in enumerate(cv.split(X_tr)):
            modelo = _ajustar(esp["clase"], params, X_tr.iloc[i_fit], y_tr.iloc[i_fit])
            maes.append(mean_absolute_error(y_tr.iloc[i_eval], modelo.predict(X_tr.iloc[i_eval])))
            trial.report(float(np.mean(maes)), step=paso)
            if trial.should_prune():
                anotar(params, maes, time.perf_counter() - t0, "podado")
                raise optuna.TrialPruned()
        anotar(params, maes, time.perf_counter() - t0, "completo")
        return float(np.mean(maes))
    return objetivo


def comparar_modelos(
    n_trials: int = 50,
    n_splits: int = 5,
    modelos: list[str] | None = None,
    timeout: float | None = None,
    ruta_datos: str | Path = RUTA_FINAL,
    ruta_logs: str | Path = RUTA_LOGS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Baselines + un estudio de Optuna por modelo (TPE + MedianPruner, semilla fija).
    La búsqueda usa TimeSeriesSplit sobre train; la mejor configuración se reentrena en todo
    train y se mide en validation. Registra cada trial y cada modelo final en el log.
    Devuelve las filas de esta corrida."""
    modelos = modelos or list(ESPACIOS)
    desconocidos = set(modelos) - set(ESPACIOS)
    if desconocidos:
        raise ValueError(f"Modelos desconocidos: {sorted(desconocidos)}. Opciones: {list(ESPACIOS)}")

    df = cargar_datos(ruta_datos)  # sin test
    X_tr, y_tr = particion(df, "train")
    X_va, y_va = particion(df, "val")
    sha = huella(ruta_datos)[:12]
    corrida = datetime.now().isoformat(timespec="seconds")
    cv = TimeSeriesSplit(n_splits=n_splits)
    filas = []

    def base_fila(modelo: str, etapa: str, estado: str, params: dict, seg: float) -> dict:
        return {
            "corrida": corrida, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "modelo": modelo, "etapa": etapa, "estado": estado,
            "hiperparametros": json.dumps(params, sort_keys=True, default=str),
            "seg_entrenamiento": round(seg, 3),
            "n_features": X_tr.shape[1], "n_train": len(X_tr), "n_val": len(X_va),
            "semilla": SEMILLA, "dataset_sha256": sha,
        }

    def anotar_val(modelo: str, etapa: str, params: dict, m_val: dict, mae_train: float, seg: float) -> None:
        # redondeo: con n_jobs=-1 el orden de suma de los árboles cambia el decimal 13
        # (ruido de punto flotante); a 6 decimales dos corridas dan exactamente lo mismo
        fila = {**base_fila(modelo, etapa, "completo", params, seg),
                "mae_val": round(m_val["mae"], 6), "rmse_val": round(m_val["rmse"], 6),
                "r2_val": round(m_val["r2"], 6), "mae_train": round(mae_train, 6)}
        registrar_experimento(fila, ruta_logs)
        filas.append(fila)

    # --- baselines (mismas filas, mismo conjunto que los modelos)
    base = predicciones_baselines(df)
    for nombre in base:
        anotar_val(nombre, "baseline", {}, _metricas(y_va, base.loc[X_va.index, nombre]),
                   _metricas(y_tr, base.loc[X_tr.index, nombre])["mae"], 0.0)
        if verbose:
            print(f"{nombre:26s} MAE val = {filas[-1]['mae_val']:.1f}")

    # --- modelos
    for nombre in modelos:
        esp = ESPACIOS[nombre]
        t_modelo = time.perf_counter()

        def anotar_cv(params: dict, maes: list[float], seg: float, estado: str, _n=nombre) -> None:
            fila = {**base_fila(_n, "optuna_cv", estado, params, seg),
                    "mae_cv": round(float(np.mean(maes)), 6),
                    "std_cv": round(float(np.std(maes)), 6), "n_folds": len(maes)}
            registrar_experimento(fila, ruta_logs)
            filas.append(fila)

        estudio = optuna.create_study(
            direction="minimize",
            study_name=f"{nombre}_{corrida}",
            sampler=optuna.samplers.TPESampler(seed=SEMILLA),
            # no poda hasta tener 5 trials completos ni antes del 2.º fold
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        estudio.optimize(_objetivo(esp, X_tr, y_tr, cv, anotar_cv),
                         n_trials=n_trials, timeout=timeout)

        # reconstruye los params de la mejor config (con el mapeo de capas del MLP)
        mejores = {**esp["fijos"], **esp["espacio"](optuna.trial.FixedTrial(estudio.best_params))}
        t0 = time.perf_counter()
        modelo = _ajustar(esp["clase"], mejores, X_tr, y_tr)
        seg = time.perf_counter() - t0
        anotar_val(nombre, "optuna_val", mejores, _metricas(y_va, modelo.predict(X_va)),
                   _metricas(y_tr, modelo.predict(X_tr))["mae"], seg)

        if verbose:
            estados = [t.state for t in estudio.trials]
            podados = sum(s == optuna.trial.TrialState.PRUNED for s in estados)
            print(f"{nombre:26s} MAE cv = {estudio.best_value:.1f}  MAE val = {filas[-1]['mae_val']:.1f}  "
                  f"({len(estados)} trials, {podados} podados, {time.perf_counter() - t_modelo:.0f} s)")

    return pd.DataFrame(filas)


def resumen(resultados: pd.DataFrame) -> pd.DataFrame:
    """Modelo final de cada estudio (y cada baseline), ordenado por MAE de validación."""
    finales = resultados[resultados["etapa"].isin(["baseline", "optuna_val"])]
    mejores = finales.loc[finales.groupby("modelo")["mae_val"].idxmin()]
    cv = (resultados[resultados["etapa"] == "optuna_cv"]
          .groupby("modelo")["mae_cv"].min().rename("mae_cv"))
    return (mejores.drop(columns="mae_cv", errors="ignore").join(cv, on="modelo")
            .sort_values("mae_val")
            [["modelo", "mae_cv", "mae_val", "rmse_val", "r2_val", "mae_train", "hiperparametros"]]
            .reset_index(drop=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=50, help="trials de Optuna por modelo (default 50)")
    ap.add_argument("--n-splits", type=int, default=5, help="folds de TimeSeriesSplit sobre train (default 5)")
    ap.add_argument("--modelos", type=str, default=None,
                    help=f"lista separada por comas (default: todos = {','.join(ESPACIOS)})")
    ap.add_argument("--timeout", type=float, default=None, help="segundos máximos por estudio")
    args = ap.parse_args()

    res = comparar_modelos(
        n_trials=args.n_trials, n_splits=args.n_splits, timeout=args.timeout,
        modelos=args.modelos.split(",") if args.modelos else None,
    )
    print("\nModelo final por estudio (CV en train, medido en validación):")
    print(resumen(res).drop(columns="hiperparametros").round(3).to_string(index=False))
    print(f"\nLog: {RUTA_LOGS}")