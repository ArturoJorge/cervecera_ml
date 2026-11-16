"""Búsqueda de hiperparámetros con Optuna para el MLP de producción y registro del mejor como versión nueva.

Parte de la config de una versión base del registro (PipelineFrio: derivadas + Preprocess + pipeline del modelo)
y busca los hiperparámetros del MLP y la config de construir_pipeline (transformador, scaler, selección):

    df_total.xlsx -> split temporal -> derivadas + Preprocess ajustado con train      (src/train_model.py)
    -> Optuna: MAE medio en TimeSeriesSplit sobre train (TPE + MedianPruner); los primeros trials son la config
       de la base y los mejores de un estudio anterior
    -> mejor trial: entrenado con train y medido en val; después entrenado con train+val y registrado

Espacios (--espacio):
    exhaustivo: el de la búsqueda exhaustiva de notebooks/modelado.ipynb (con el que salieron v2.0.1 y v2.0.3)
    refinado:   acotado con el análisis del estudio de v2.0.3 (con el que salió v2.0.2); además busca el umbral del filtro por
                correlación de Preprocess (0.9 / 0.95 / 0.98 / 1.0 = sin filtro)

Uso (env cervecera_env):
    python src/main.py                                     # refinado, base v2.0.3 -> mlp v2.0.4 (45 min)
    python src/main.py --trials 50 --timeout 600 --version v2.0.5
    python src/main.py --version v2.0.5 --estudio "mlp_v2.0.1_2026-11-15T19:33:01"   # registra un estudio guardado

Ojo: los nombres de los estudios no coinciden con la versión registrada (v2.0.2 y v2.0.3 se intercambiaron para que
la más nueva sea la mejor): v2.0.3 salió de "mlp_v2.0.1_2026-11-15T19:33:01" y v2.0.2 de "mlp_v2.0.3_2026-11-15T20:21:18".

No correr dos a la vez con la misma --version: la que termine segunda no va a poder registrarse.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import train_model as tm
from feature_selection import SEMILLA, memoria
from preprocess import COL_FECHA, ROOT, RUTA_LIMPIO, TARGET, ConfigCorrelacion, construir_dataset

MODELO = "mlp"
BASE = "v2.0.3"
VERSION = "v2.0.4"
ESPACIO = "refinado"
ENCOLAR_DE = "mlp_v2.0.1_2026-11-15T19:33:01"   # estudio de v2.0.3: sus 2 mejores trials arrancan el estudio nuevo
N_TRIALS = 150
TIMEOUT = 45 * 60   # segundos (None = sin límite)
N_SPLITS = 5
RUTA_OPTUNA_DB = ROOT / "results" / "optuna.db"   # mismo storage que el notebook: optuna-dashboard lo abre
STORAGE = f"sqlite:///{RUTA_OPTUNA_DB.as_posix()}"

# Sin early_stopping: scikit-learn lo haría con un 10 % ALEATORIO de train (mezcla pasado y futuro)
FIJOS = {"random_state": SEMILLA, "solver": "adam", "max_iter": 1000, "n_iter_no_change": 20, "early_stopping": False}
ARQUITECTURA = ("hidden_layer_sizes", "activation")   # el resto va a "hiperparametros" en el registro
UMBRALES_CORR = [0.9, 0.95, 0.98, 1.0]                # 1.0 = sin filtro (ninguna |corr| es > 1)


def capas_mlp(n_capas: int, neuronas: int, forma: str) -> tuple[int, ...]:
    """constante: 64-64-64 | embudo: cada capa la mitad de la anterior, mínimo 8 (128-64-32-16)."""
    return tuple(neuronas if forma == "constante" else max(8, neuronas // 2 ** i) for i in range(n_capas))


# ------------------------------------------------------------------ espacios de búsqueda
# Cada uno devuelve (parámetros del MLP, config de construir_pipeline, umbral de correlación o None = el de la base).
# Con un FixedTrial reconstruyen un trial guardado. Usan los mismos nombres de parámetros.

def _mlp(t: optuna.Trial, n_capas, neuronas, activaciones, alpha, lr, batch) -> dict:
    return {
        **FIJOS,
        "hidden_layer_sizes": capas_mlp(t.suggest_int("n_capas", *n_capas), t.suggest_categorical("neuronas", neuronas),
                                        t.suggest_categorical("forma", ["constante", "embudo"])),
        "activation": t.suggest_categorical("activation", activaciones),
        "alpha": t.suggest_float("alpha", *alpha, log=True),
        "learning_rate_init": t.suggest_float("learning_rate_init", *lr, log=True),
        "batch_size": t.suggest_categorical("batch_size", batch),
    }


def _pipeline(t: optuna.Trial, transformadores, scalers, selecciones, metodos, k, umbral_red, frac_rfe,
              n_rfe_min) -> dict:
    cfg = {
        "transformador": (transformadores[0] if len(transformadores) == 1
                          else t.suggest_categorical("transformador", transformadores)),
        "scaler": t.suggest_categorical("scaler", scalers),
        "seleccion": t.suggest_categorical("seleccion", selecciones),
    }
    if cfg["seleccion"] in ("filtro", "filtro+rfe"):
        cfg["filtro_metodo"] = t.suggest_categorical("filtro_metodo", metodos)
        cfg["k"] = t.suggest_int("k", k[0], k[1], step=k[2] if len(k) > 2 else 1)   # k = (mín, máx[, paso])
        cfg["umbral_redundancia"] = t.suggest_float("umbral_redundancia", *umbral_red)
    if cfg["seleccion"] == "filtro+rfe":
        cfg["n_rfe"] = max(n_rfe_min, int(cfg["k"] * t.suggest_float("frac_rfe", *frac_rfe)))
    if cfg["seleccion"] == "lasso":
        cfg["alpha_sel"] = t.suggest_float("alpha_sel", 1e-1, 1e3, log=True)
    return cfg


def sugerir_exhaustivo(t: optuna.Trial) -> tuple[dict, dict, float | None]:
    """El de la búsqueda exhaustiva del MLP en notebooks/modelado.ipynb."""
    params = _mlp(t, (1, 4), [16, 32, 64, 128, 256], ["relu", "tanh"], (1e-4, 100.0), (1e-4, 1e-2),
                  [16, 32, 64, 128, 256])
    cfg = _pipeline(t, ["yeo-johnson", "quantile", "ninguno"], ["standard", "robust", "minmax", "ninguno"],
                    ["filtro", "filtro+rfe", "lasso", "ninguna"], ["pearson", "spearman", "mutual_info"],
                    (5, 200, 5), (0.80, 1.0), (0.2, 1.0), n_rfe_min=3)
    return params, cfg, None


def sugerir_refinado(t: optuna.Trial) -> tuple[dict, dict, float | None]:
    """Acotado con el estudio de v2.0.3: fuera relu, transformadores, scaler standard/ninguno, selección
    lasso/ninguna, mutual_info y batch 16/32/256 (nunca dieron buenos resultados); arquitectura abierta hacia
    arriba (el óptimo estaba en 4 capas × 256) y selección abierta hacia abajo (estaba en k=5, n_rfe=3 = mínimos)."""
    params = _mlp(t, (3, 5), [64, 128, 256, 512], ["tanh"], (0.05, 20.0), (5e-5, 2e-3), [64, 128])
    cfg = _pipeline(t, ["ninguno"], ["robust", "minmax"], ["filtro", "filtro+rfe"], ["pearson", "spearman"],
                    (1, 30), (0.85, 1.0), (0.05, 1.0), n_rfe_min=1)
    return params, cfg, t.suggest_categorical("umbral_corr", UMBRALES_CORR)


ESPACIOS = {"exhaustivo": sugerir_exhaustivo, "refinado": sugerir_refinado}


def reconstruir(estudio: optuna.Study, trial: optuna.trial.FrozenTrial) -> tuple[dict, dict, float | None]:
    """(params, cfg, umbral) de un trial, con el espacio con el que se hizo (los estudios viejos: exhaustivo)."""
    return ESPACIOS[estudio.user_attrs.get("espacio", "exhaustivo")](optuna.trial.FixedTrial(trial.params))


def a_parametros(params: dict, cfg: dict, umbral: float | None = None) -> dict | None:
    """(params, cfg) -> parámetros de Optuna, para encolar un trial. None si la arquitectura no es constante/embudo."""
    capas = tuple(params["hidden_layer_sizes"])
    forma = next((f for f in ("constante", "embudo") if capas_mlp(len(capas), capas[0], f) == capas), None)
    if forma is None:
        return None
    p = {k: v for k, v in cfg.items() if k not in ("n_rfe", "externas_siempre")}
    if "n_rfe" in cfg:
        p["frac_rfe"] = (cfg["n_rfe"] + 0.5) / cfg["k"]   # int(k * frac) vuelve a dar n_rfe (con cualquier n_rfe_min)
    p |= {"n_capas": len(capas), "neuronas": capas[0], "forma": forma,
          **{k: params[k] for k in ("activation", "alpha", "learning_rate_init", "batch_size")}}
    if umbral is not None:
        p["umbral_corr"] = umbral
    return p


def para_encolar(entrada: dict, umbral_base: float | None, refinado: bool, de: str | None = ENCOLAR_DE,
                 n: int = 2) -> list[dict]:
    """Config de la base + los `n` mejores trials completos del estudio `de` (sin repetidos)."""
    u = umbral_base if refinado else None
    candidatos = [a_parametros(tm.params_modelo(entrada), tm.cfg_pipeline(entrada), u)]
    if de:
        previo = optuna.load_study(study_name=de, storage=STORAGE)
        completos = sorted((t for t in previo.trials if t.state == optuna.trial.TrialState.COMPLETE),
                           key=lambda t: t.value)
        for t in completos[:n]:
            params, cfg, u_t = reconstruir(previo, t)
            candidatos.append(a_parametros(params, cfg, u_t if u_t is not None else u))
    unicos = []
    for c in candidatos:
        if c is not None and c not in unicos:
            unicos.append(c)
    return unicos


# ------------------------------------------------------------------ datos y estudio
def datos_train(base: str, umbrales: list[float]) -> tuple[dict, pd.Series, dict, float | None]:
    """({umbral: X}, y, entrada de la base, umbral de la base). X = variables de train tal como las ve el modelo
    (derivadas + Preprocess ajustado con train). Una X por umbral de correlación, armada una sola vez; sin
    umbrales, solo la de la base (clave None). Val y test no se tocan."""
    cfg = tm.cargar_config(MODELO, base)
    pipe, train, _, _ = tm._pipeline_y_datos(cfg)
    umbral_base = pipe.preprocess.correlacion.umbral if pipe.preprocess.correlacion else None
    Xs = {}
    for u in umbrales or [None]:
        p = copy.deepcopy(pipe)
        if u is not None:
            p.preprocess.correlacion = ConfigCorrelacion(umbral=u)
        p.ajustar_preprocess(train)
        Xs[u] = p.transformar(train)
    X = next(iter(Xs.values()))
    y = train.set_index(COL_FECHA)[TARGET].reindex(X.index)
    ok = y.notna()   # el último día no tiene "frío de mañana"
    return {u: X[ok] for u, X in Xs.items()}, y[ok], cfg["registro"], umbral_base


def objetivo(Xs: dict, y: pd.Series, cv: TimeSeriesSplit, sugerir):
    """MAE medio en los folds; reporta el acumulado en cada fold para que el pruner corte los trials malos.
    Una config inválida (p. ej. la selección deja 0 columnas) se poda."""
    def f(trial: optuna.Trial) -> float:
        params, cfg, umbral = sugerir(trial)
        X = Xs[umbral]
        trial.set_user_attr("n_columnas", X.shape[1])
        maes = []
        for paso, (i_fit, i_eval) in enumerate(cv.split(X)):
            try:
                with tm._sin_avisos():
                    # Ridge para el RFE: el MLP no tiene coef_ (igual que crear_modelo)
                    pipe = tm.construir_pipeline(tm.crear_mlp(**params), X.columns, cfg, Ridge(), memory=memoria)
                    pred = pipe.fit(X.iloc[i_fit], y.iloc[i_fit]).predict(X.iloc[i_eval])
            except ValueError as e:
                trial.set_user_attr("error", str(e)[:200])
                raise optuna.TrialPruned() from e
            maes.append(mean_absolute_error(y.iloc[i_eval], pred))
            trial.report(float(np.mean(maes)), step=paso)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("std_cv", float(np.std(maes)))
        return float(np.mean(maes))
    return f


def estudiar(Xs: dict, y: pd.Series, nombre: str, espacio: str = ESPACIO, encolar: list[dict] | None = None,
             n_trials: int = N_TRIALS, timeout: float | None = TIMEOUT, storage: str | None = STORAGE) -> optuna.Study:
    """Estudio de Optuna (TPE multivariado + MedianPruner, semilla fija) sobre TimeSeriesSplit de train."""
    RUTA_OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    # test_size fijo: la ventana de entrenamiento más chica es la mitad de train
    cv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=len(y) // (2 * N_SPLITS))
    estudio = optuna.create_study(
        direction="minimize", study_name=nombre, storage=storage,
        # multivariate + group: el espacio es condicional (cada selección tiene sus propios parámetros)
        sampler=optuna.samplers.TPESampler(seed=SEMILLA, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    estudio.set_user_attr("espacio", espacio)   # para reconstruir los trials después (reconstruir / --estudio)
    for params in encolar or []:
        estudio.enqueue_trial(params)
    # barra de Optuna (tqdm): trials hechos / n_trials y mejor MAE cv hasta el momento
    estudio.optimize(objetivo(Xs, y, cv, ESPACIOS[espacio]), n_trials=n_trials, timeout=timeout,
                     show_progress_bar=True)
    return estudio


def resumen(estudio: optuna.Study, base: str) -> None:
    trials = estudio.trials
    estados = pd.Series([t.state.name for t in trials]).value_counts().to_dict()
    primero = trials[0].value
    print(f"{len(trials)} trials {estados} | MAE cv mejor {estudio.best_value:.1f} (trial {estudio.best_trial.number})"
          f" | config de {base}: {'podado' if primero is None else f'{primero:.1f}'}")
    completos = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and "umbral_corr" in t.params]
    if completos:
        por_umbral = pd.DataFrame([{"umbral_corr": t.params["umbral_corr"], "mae_cv": t.value} for t in completos])
        print("MAE cv por umbral de correlación (completos):")
        print(por_umbral.groupby("umbral_corr")["mae_cv"].agg(["min", "median", "count"]).round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna sobre el MLP de producción y registro del mejor trial.")
    parser.add_argument("--base", default=BASE, help="versión de la que se toman datos, Preprocess y 1.er trial")
    parser.add_argument("--version", default=VERSION, help="versión con la que se registra el mejor")
    parser.add_argument("--espacio", default=ESPACIO, choices=list(ESPACIOS))
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="segundos (0 = sin límite)")
    parser.add_argument("--estudio", help="nombre de un estudio ya guardado en results/optuna.db: no se optimiza, "
                                          "se registra su mejor trial")
    args = parser.parse_args()
    version = tm._version_libre(MODELO, args.version, tm.RUTA_REGISTRO_MODELOS)   # antes del estudio: falla rápido

    if not RUTA_LIMPIO.exists():   # crudos -> data/clean/df_total.xlsx
        construir_dataset(guardar=RUTA_LIMPIO)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    estudio = optuna.load_study(study_name=args.estudio, storage=STORAGE) if args.estudio else None
    espacio = estudio.user_attrs.get("espacio", "exhaustivo") if estudio else args.espacio
    Xs, y, entrada, umbral_base = datos_train(args.base, UMBRALES_CORR if espacio == "refinado" else [])

    if estudio:   # p. ej. una corrida que optimizó bien pero no pudo registrar
        nombre, corrida = estudio.study_name, estudio.trials[0].datetime_start.isoformat(timespec="seconds")
        print(f"Estudio guardado {nombre} (espacio {espacio}, config de {MODELO} {args.base})")
    else:
        corrida = datetime.now().isoformat(timespec="seconds")
        nombre = f"{MODELO}_{version}_{corrida}"
        print(f"Estudio {nombre} (espacio {espacio}, config de {MODELO} {args.base}): "
              + ", ".join(f"{u}: {X.shape}" for u, X in Xs.items()))
        encolar = para_encolar(entrada, umbral_base, espacio == "refinado")
        estudio = estudiar(Xs, y, nombre, espacio, encolar, args.trials, args.timeout or None)
    resumen(estudio, args.base)

    params, cfg, umbral = reconstruir(estudio, estudio.best_trial)
    config = {
        "arquitectura": {k: params[k] for k in ARQUITECTURA},
        "hiperparametros": {k: v for k, v in params.items() if k not in ARQUITECTURA},
        "preprocesamiento": {**cfg, "externas_siempre": tm.columnas_externas(Xs[umbral].columns)},
        "estudio_optuna": nombre,
        "corrida": corrida,
    }
    # None -> el filtro de la base; si no, el umbral elegido por el estudio
    correlacion = None if umbral is None else {"umbral": umbral, "metodo": "spearman"}
    print("Mejor:", config["arquitectura"], {k: v for k, v in params.items() if k not in FIJOS}, cfg,
          f"umbral_corr={umbral if umbral is not None else umbral_base}")

    # 1) mejor config entrenada con train y medida en val (val no participó del estudio)
    _, medida = tm.entrenar(MODELO, base=args.base, datos="train", config=config, correlacion=correlacion)
    extra = {"mae_cv": estudio.best_value, "std_cv": estudio.best_trial.user_attrs.get("std_cv"),
             **{k: v for k, v in medida["metricas"].items() if k.endswith("_val")}}
    # 2) producción: train+val, se guarda y se registra
    tm.entrenar(MODELO, version, args.base, "train+val", config=config, correlacion=correlacion, metricas_extra=extra)


if __name__ == "__main__":
    main()
