"""Estudio de Optuna (TPE + poda) que configura TODO el camino hasta la predicción con XGBoost:
variables derivadas -> Preprocess (inválidos, outliers, faltantes) -> pipeline (transformador, scaler,
selección de variables) -> hiperparámetros de XGBoost.

Cada trial se mide con TimeSeriesSplit sobre train (MAE medio). Después de cada fold se reporta el MAE
acumulado y el pruner corta los trials que no prometen. Al terminar, la mejor configuración se entrena con
todo train, se mide en validation, se guarda en models/xgboost_{version}.joblib (un PipelineFrio, lo carga
src/predict.py) y se registra en models/model_registry.json. Todos los trials y el modelo final quedan en
results/experiment_logs.csv. Test no se usa nunca.

Uso:
    python src/main.py                               # 100 trials o 1 hora, lo que pase primero
    python src/main.py --trials 300 --timeout 7200
    python src/main.py --version v1.0.0              # fijar la versión en vez de calcularla
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost.core import XGBoostError

from feature_selection import SEMILLA
from preprocess import (COL_FECHA, ROOT, RUTA_LIMPIO, TARGET, ConfigDerivadas, ConfigFaltantes, ConfigInvalidos,
                        ConfigOutliers, ConfigSplit, Preprocess, cargar_dataset_limpio, cargar_registro,
                        guardar_registro, huella)
from train_model import (RUTA_MODELOS, RUTA_REGISTRO_MODELOS, TIPO_PIPELINE, PipelineFrio, _entorno, _metricas,
                         _version_libre)

MODELO = "xgboost"
N_TRIALS = 150
TIMEOUT = 60 * 60      # segundos
N_SPLITS = 5           # folds de TimeSeriesSplit sobre train
RUTA_LOGS = ROOT / "results" / "experiment_logs.csv"
RUTA_OPTUNA_DB = ROOT / "results" / "optuna.db"   # estudios persistidos: optuna-dashboard la puede abrir

# mismas columnas que notebooks/modelado.ipynb
COLUMNAS_LOG = [
    "corrida", "timestamp", "modelo", "estudio", "etapa", "estado", "hiperparametros", "preprocesamiento",
    "mae_cv", "std_cv", "n_folds", "mae_val", "rmse_val", "r2_val", "mae_train",
    "seg_entrenamiento", "n_features", "n_train", "n_val", "semilla", "dataset_sha256",
]
FIJOS_XGB = {"random_state": SEMILLA, "n_jobs": -1, "tree_method": "hist"}
_ARQUITECTURA = ("n_estimators", "max_depth")   # como en el notebook: lo demás son hiperparámetros

# Optuna no acepta listas como opciones categóricas: se elige un nombre y se traduce.
_LAGS = {"ninguno": [], "1": [1], "1-7": [1, 7], "1-2-7": [1, 2, 7], "1-2-3-7": [1, 2, 3, 7], "1-2-7-14": [1, 2, 7, 14]}
_VENTANAS = {"ninguna": [], "7": [7], "3-7": [3, 7], "7-14": [7, 14], "3-7-14": [3, 7, 14], "3-7-14-28": [3, 7, 14, 28]}


# =============================================================================
# 1. Espacio de búsqueda
# =============================================================================

def espacio_derivadas(t: optuna.Trial) -> ConfigDerivadas:
    base = ConfigDerivadas()
    return ConfigDerivadas(
        lags=_LAGS[t.suggest_categorical("lags", list(_LAGS))],
        ventanas=_VENTANAS[t.suggest_categorical("ventanas", list(_VENTANAS))],
        ratios=base.ratios if t.suggest_categorical("ratios", [True, False]) else [],
        meta=t.suggest_categorical("cumplimiento_meta", [True, False]),
        interacciones=base.interacciones if t.suggest_categorical("interacciones", [True, False]) else [],
    )


def espacio_preprocess(t: optuna.Trial) -> Preprocess:
    """Inválidos, outliers y faltantes. Sin la acción "delete": borra días y rompe los rezagos.
    Transformador y scaler van en el pipeline del modelo (espacio_pipeline)."""
    invalidos = outliers = None
    accion = t.suggest_categorical("invalidos_accion", ["nan", "clip", "ninguno"])
    if accion != "ninguno":
        invalidos = ConfigInvalidos(
            accion=accion,
            no_negativas=t.suggest_categorical("invalidos_no_negativas", ["ee", "all"]),
            max_abs=t.suggest_categorical("invalidos_max_abs", [1e5, 1e6, 1e7]),
        )
    metodo = t.suggest_categorical("outliers_metodo", ["iqr", "zscore_mod", "percentil", "ninguno"])
    if metodo != "ninguno":
        outliers = ConfigOutliers(metodo=metodo, accion=t.suggest_categorical("outliers_accion", ["clip", "nan"]))
        if metodo == "iqr":
            outliers.k_iqr = t.suggest_float("outliers_k_iqr", 1.5, 4.0)
        elif metodo == "zscore_mod":
            outliers.umbral_zmod = t.suggest_float("outliers_umbral_zmod", 2.5, 6.0)
        else:
            cola = t.suggest_float("outliers_cola", 0.001, 0.05, log=True)   # fracción recortada de cada lado
            outliers.percentiles = (cola, 1 - cola)
    faltantes = ConfigFaltantes(estrategia=t.suggest_categorical("faltantes", ["median", "mean", "interpolate", "knn"]))
    if faltantes.estrategia == "knn":
        faltantes.n_vecinos = t.suggest_int("faltantes_n_vecinos", 2, 15)
    if t.suggest_categorical("descartar_columnas_con_nulos", [True, False]):
        faltantes.max_frac_nulos_col = t.suggest_float("max_frac_nulos_col", 0.2, 0.95)
    return Preprocess(invalidos=invalidos, outliers=outliers, faltantes=faltantes,
                      transformer=None, scaler=None, split=ConfigSplit())


def espacio_pipeline(t: optuna.Trial) -> dict:
    """Configuración de train_model.construir_pipeline (igual que espacio_prep del notebook)."""
    cfg = {
        "transformador": t.suggest_categorical("transformador", ["yeo-johnson", "quantile", "ninguno"]),
        "scaler": t.suggest_categorical("scaler", ["standard", "robust", "minmax", "ninguno"]),
        "seleccion": t.suggest_categorical("seleccion", ["filtro", "filtro+rfe", "lasso", "ninguna"]),
    }
    if cfg["seleccion"] in ("filtro", "filtro+rfe"):
        cfg["filtro_metodo"] = t.suggest_categorical("filtro_metodo", ["pearson", "spearman", "mutual_info"])
        cfg["k"] = t.suggest_int("k", 5, 200, step=5)
        cfg["umbral_redundancia"] = t.suggest_float("umbral_redundancia", 0.80, 1.0)   # 1.0 = sin filtro de redundancia
    if cfg["seleccion"] == "filtro+rfe":
        cfg["n_rfe"] = max(3, int(cfg["k"] * t.suggest_float("frac_rfe", 0.2, 1.0)))
    if cfg["seleccion"] == "lasso":
        cfg["alpha_sel"] = t.suggest_float("alpha_sel", 1e-1, 1e3, log=True)
    return cfg


def espacio_xgb(t: optuna.Trial) -> dict:
    return {
        **FIJOS_XGB,
        "n_estimators": t.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
        "gamma": t.suggest_float("gamma", 1e-3, 10.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 100.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-1, 100.0, log=True),
    }


def sugerir(t: optuna.Trial) -> PipelineFrio:
    """PipelineFrio sin entrenar con todo lo que elige el trial. Con un FixedTrial reconstruye el mejor."""
    derivadas, preprocess = espacio_derivadas(t), espacio_preprocess(t)
    return PipelineFrio(MODELO, espacio_xgb(t), espacio_pipeline(t), derivadas, preprocess)


def config_preprocesamiento(pipe: PipelineFrio) -> dict:
    """Todo lo que no es hiperparámetro del modelo, en formato JSON (va al log y al registro)."""
    resumen = pipe.preprocess.resumen()["config"]
    return json.loads(json.dumps({
        "derivadas": asdict(pipe.cfg_derivadas),
        "preprocess": {k: resumen[k] for k in ("invalidos", "outliers", "faltantes")},
        "pipeline": pipe.cfg_pipeline,
    }, default=str))


# =============================================================================
# 2. Log de experimentos
# =============================================================================

def registrar_experimento(fila: dict, ruta: str | Path = RUTA_LOGS) -> None:
    """Agrega una fila a results/experiment_logs.csv (lo crea si no existe), con las columnas de COLUMNAS_LOG."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nueva = pd.DataFrame([{c: fila.get(c, np.nan) for c in COLUMNAS_LOG}])
    if not ruta.exists():
        nueva.to_csv(ruta, index=False)
    elif list(pd.read_csv(ruta, nrows=0).columns) == COLUMNAS_LOG:
        nueva.to_csv(ruta, mode="a", header=False, index=False)
    else:   # log con otras columnas: se completa sin perder las filas viejas
        previo = pd.read_csv(ruta)
        extra = [c for c in previo.columns if c not in COLUMNAS_LOG]
        pd.concat([previo, nueva], ignore_index=True).reindex(columns=COLUMNAS_LOG + extra).to_csv(ruta, index=False)


# =============================================================================
# 3. Estudio
# =============================================================================

def _predecir(pipe: PipelineFrio, historia: pd.DataFrame, fechas: pd.Series) -> pd.Series:
    """Predicción para `fechas`; `historia` incluye los días anteriores, que hacen falta para los rezagos."""
    return pipe.predict(historia).loc[pd.to_datetime(fechas)]


def _objetivo(train: pd.DataFrame, cv: TimeSeriesSplit, anotar):
    """MAE medio sobre los folds. Reporta el MAE acumulado después de cada fold: si el pruner decide que el
    trial no promete, se corta ahí. Una config inválida (p. ej. la selección deja 0 columnas) también se poda."""
    def objetivo(trial: optuna.Trial) -> float:
        pipe = sugerir(trial)
        maes: list[float] = []
        n_feat = None
        t0 = time.perf_counter()
        for paso, (i_fit, i_eval) in enumerate(cv.split(train)):
            try:
                pipe.fit(train.iloc[i_fit])
                evaluar = train.iloc[i_eval]
                pred = _predecir(pipe, train.iloc[: i_eval[-1] + 1], evaluar[COL_FECHA])
            except (ValueError, XGBoostError) as e:
                trial.set_user_attr("error", str(e)[:200])
                anotar(pipe, maes or [np.nan], time.perf_counter() - t0, "podado", n_feat)
                raise optuna.TrialPruned() from e
            n_feat = len(pipe.columnas_)
            trial.set_user_attr("n_features", n_feat)
            maes.append(mean_absolute_error(evaluar[TARGET], pred))
            trial.report(float(np.mean(maes)), step=paso)
            if trial.should_prune():
                anotar(pipe, maes, time.perf_counter() - t0, "podado", n_feat)
                raise optuna.TrialPruned()
        trial.set_user_attr("std_cv", float(np.std(maes)))
        anotar(pipe, maes, time.perf_counter() - t0, "completo", n_feat)
        return float(np.mean(maes))
    return objetivo


def version_siguiente(previa: dict | None, nueva: dict) -> str | None:
    """vMAJOR.MINOR.PATCH según qué cambió respecto de `previa` (misma regla que el notebook):
    datos -> MAJOR, preprocesamiento o arquitectura -> MINOR, hiperparámetros -> PATCH. None si no cambió nada."""
    if previa is None:
        return "v0.0.0"
    mayor, menor, parche = map(int, previa["version"].removeprefix("v").split("."))
    if previa["dataset"]["sha256"] != nueva["dataset"]["sha256"]:
        return f"v{mayor + 1}.0.0"
    if previa["preprocesamiento"] != nueva["preprocesamiento"] or previa["arquitectura"] != nueva["arquitectura"]:
        return f"v{mayor}.{menor + 1}.0"
    if previa["hiperparametros"] != nueva["hiperparametros"]:
        return f"v{mayor}.{menor}.{parche + 1}"
    return None


def estudiar(n_trials: int = N_TRIALS, timeout: float | None = TIMEOUT, n_splits: int = N_SPLITS,
             version: str | None = None, ruta_datos: str | Path = RUTA_LIMPIO, ruta_logs: str | Path = RUTA_LOGS,
             registro: str | Path = RUTA_REGISTRO_MODELOS, carpeta: str | Path = RUTA_MODELOS,
             ruta_db: str | Path = RUTA_OPTUNA_DB) -> tuple[optuna.Study, PipelineFrio, dict | None]:
    """Corre el estudio, entrena la mejor config con train, la mide en val, la guarda y la registra.
    Devuelve (estudio, pipeline final, entrada del registro o None si no se registró)."""
    if version is not None:
        version = _version_libre(MODELO, version, registro)   # antes de optimizar: falla rápido
    ruta_datos = Path(ruta_datos)
    train, val, _ = Preprocess(split=ConfigSplit()).train_test_split(cargar_dataset_limpio(ruta_datos))  # sin test
    train, val = train.dropna(subset=[TARGET]), val.dropna(subset=[TARGET])
    sha = huella(ruta_datos)
    corrida = datetime.now().isoformat(timespec="seconds")
    nombre_estudio = f"{MODELO}_main_{corrida}"
    # test_size fijo: la ventana de entrenamiento más chica es la mitad de train
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=len(train) // (2 * n_splits))
    print(f"train {len(train)} días ({train[COL_FECHA].min():%Y-%m-%d} a {train[COL_FECHA].max():%Y-%m-%d}) | "
          f"val {len(val)} días | {n_splits} folds | estudio {nombre_estudio}")

    def anotar(pipe, etapa, estado, seg, n_feat, **metricas) -> dict:
        fila = {
            "corrida": corrida, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "modelo": MODELO, "estudio": nombre_estudio, "etapa": etapa, "estado": estado,
            "hiperparametros": json.dumps(pipe.params_modelo, sort_keys=True, default=str),
            "preprocesamiento": json.dumps(config_preprocesamiento(pipe), sort_keys=True),
            "seg_entrenamiento": round(seg, 3), "n_features": n_feat,
            "n_train": len(train), "n_val": len(val), "semilla": SEMILLA, "dataset_sha256": sha[:12],
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metricas.items()},
        }
        registrar_experimento(fila, ruta_logs)
        return fila

    def anotar_cv(pipe, maes, seg, estado, n_feat):
        todo_nan = bool(np.all(np.isnan(maes)))
        anotar(pipe, "optuna_cv", estado, seg, n_feat,
               mae_cv=np.nan if todo_nan else float(np.nanmean(maes)),
               std_cv=np.nan if todo_nan else float(np.nanstd(maes)),
               n_folds=int(np.sum(~np.isnan(maes))))

    Path(ruta_db).parent.mkdir(parents=True, exist_ok=True)
    estudio = optuna.create_study(
        direction="minimize", study_name=nombre_estudio, storage=f"sqlite:///{Path(ruta_db).as_posix()}",
        # multivariate + group: el espacio es condicional (cada método tiene sus propios parámetros)
        sampler=optuna.samplers.TPESampler(seed=SEMILLA, multivariate=True, group=True),
        # no poda hasta tener 10 trials completos ni antes del 2.º fold
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    t0 = time.perf_counter()
    estudio.optimize(_objetivo(train, cv, anotar_cv), n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    estados = [t.state for t in estudio.trials]
    completos = sum(s == optuna.trial.TrialState.COMPLETE for s in estados)
    podados = sum(s == optuna.trial.TrialState.PRUNED for s in estados)
    print(f"{len(estados)} trials ({completos} completos, {podados} podados) en {time.perf_counter() - t0:.0f} s")
    if not completos:
        raise RuntimeError("Ningún trial terminó: no hay modelo para guardar")
    print(f"Mejor trial #{estudio.best_trial.number}: MAE cv = {estudio.best_value:.1f}")

    # --- mejor config -> todo train -> validation
    pipe = sugerir(optuna.trial.FixedTrial(estudio.best_params))
    t0 = time.perf_counter()
    pipe.fit(train)
    seg = time.perf_counter() - t0
    m_val = _metricas(val.set_index(COL_FECHA)[TARGET], _predecir(pipe, pd.concat([train, val]), val[COL_FECHA]))
    m_train = _metricas(train.set_index(COL_FECHA)[TARGET], pipe.predict(train))
    # persistencia: el frío de mañana se predice con el de hoy
    mae_persistencia = float(mean_absolute_error(val[TARGET], val["frio (kw)"]))
    fila = anotar(pipe, "optuna_val", "completo", seg, len(pipe.columnas_),
                  mae_cv=estudio.best_value, std_cv=estudio.best_trial.user_attrs.get("std_cv", np.nan),
                  n_folds=n_splits, mae_val=m_val["mae"], rmse_val=m_val["rmse"], r2_val=m_val["r2"],
                  mae_train=m_train["mae"])
    print(f"Validation: MAE {m_val['mae']:.1f} | RMSE {m_val['rmse']:.1f} | R2 {m_val['r2']:.3f} | "
          f"persistencia MAE {mae_persistencia:.1f} | {len(pipe.columnas_)} variables")

    # --- registro
    params = json.loads(json.dumps(pipe.params_modelo, sort_keys=True, default=str))
    entradas = cargar_registro(registro)
    nueva = {
        "dataset": {"archivo": ruta_datos.resolve().relative_to(ROOT).as_posix() if ruta_datos.resolve().is_relative_to(ROOT)
                    else str(ruta_datos), "sha256": sha},
        "arquitectura": {k: params[k] for k in _ARQUITECTURA},
        "hiperparametros": {k: v for k, v in params.items() if k not in _ARQUITECTURA},
        "preprocesamiento": config_preprocesamiento(pipe),
    }
    previa = next((e for e in reversed(entradas) if e["modelo"] == MODELO), None)
    version = version or version_siguiente(previa, nueva)
    if version is None:
        print(f"{MODELO}: sin cambios respecto de {previa['version']}, no se registra.")
        return estudio, pipe, None

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{MODELO}_{version}.joblib"
    joblib.dump(pipe, ruta, compress=3)
    archivo = ruta.resolve().relative_to(ROOT).as_posix() if ruta.resolve().is_relative_to(ROOT) else str(ruta)
    entrada = {
        "modelo": MODELO, "version": version, "tipo": TIPO_PIPELINE,
        "archivo": archivo, "sha256": huella(ruta),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "version_previa": previa["version"] if previa else None,
        "estudio_optuna": nombre_estudio, "corrida": corrida,
        **nueva,
        "metricas": {
            **{k: fila.get(k) for k in ("mae_cv", "std_cv", "mae_val", "rmse_val", "r2_val", "mae_train")},
            "mejora_val_vs_persistencia_%": round(100 * (1 - m_val["mae"] / mae_persistencia), 3),
        },
        "entrenado_con": {"split": "train", "n_filas": len(train),
                          "desde": f"{train[COL_FECHA].min():%Y-%m-%d}", "hasta": f"{train[COL_FECHA].max():%Y-%m-%d}"},
        "n_features": len(pipe.columnas_), "columnas": pipe.columnas_, "semilla": SEMILLA,
        "entorno": {**_entorno(), "optuna": optuna.__version__, "xgboost": __import__("xgboost").__version__},
        "cargar_con": f"sys.path.insert(0, 'src'); joblib.load('{archivo}').predecir_manana(df)",
    }
    guardar_registro(entradas + [entrada], registro)
    print(f"Guardado {archivo} y registrado {MODELO} {version} en {Path(registro).name}")
    return estudio, pipe, entrada


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudio de Optuna con poda: preprocesamiento + XGBoost.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="cantidad máxima de trials (por defecto %(default)s)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT,
                        help="segundos máximos del estudio; 0 = sin límite (por defecto %(default)s)")
    parser.add_argument("--folds", type=int, default=N_SPLITS, help="folds de TimeSeriesSplit (por defecto %(default)s)")
    parser.add_argument("--version", help="versión con la que se registra (por defecto se calcula sola)")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudiar(args.trials, args.timeout or None, args.folds, args.version)


if __name__ == "__main__":
    main()
