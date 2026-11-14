"""Entrenamiento del modelo final y pipeline completo para producción.

Cadena de configuración (nada está escrito a mano acá):
    models/model_registry.json (modelo + versión)  -> hiperparámetros, arquitectura y pipeline del modelo
    -> dataset.sha256 -> data/logs/data_lineage.json -> variables derivadas, split y Preprocess
    -> archivo de entrada (data/clean/df_total.xlsx)

`PipelineFrio` va del dato diario limpio (fecha + columnas de planta y clima) a la predicción del frío de mañana:
    agregar_derivadas -> Preprocess (inválidos, outliers, faltantes) -> construir_pipeline (scaler, selección, modelo)

Uso:
    python src/train_model.py --version v1.0.0                 # config del mejor mlp registrado, entrenado con
                                                               # train+val y registrado como v1.0.0
    python src/train_model.py --version v1.0.1 --base v0.1.0   # elegir de qué versión se toma la config
    python src/train_model.py                                  # pide la versión por teclado
    python src/train_model.py --datos train --base v0.1.0      # solo verifica: reproduce v0.1.0 y lo mide en
                                                               # val (no guarda ni registra nada)

Cargar el pipeline desde otro script:
    sys.path.insert(0, "src"); pipe = joblib.load("models/mlp_v1.0.0.joblib")
    pipe.predecir_manana(df_ultimos_dias)          # df con al menos 15 días, como df_total.xlsx
"""
from __future__ import annotations

import argparse
import copy
import platform
import re
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

from feature_selection import SEMILLA, SelectorFiltro
from preprocess import (COL_FECHA, COLS_CICLICAS, COLS_SIN_LIMITES, ROOT, RUTA_LINEAGE, TARGET,
                        ConfigDerivadas, ConfigFaltantes, ConfigInvalidos, ConfigOutliers, ConfigSplit,
                        Preprocess, _clave, agregar_derivadas, cargar_dataset_limpio, cargar_registro,
                        guardar_registro, huella)

RUTA_MODELOS = ROOT / "models"
RUTA_REGISTRO_MODELOS = RUTA_MODELOS / "model_registry.json"
MODELO = "mlp"
TIPO_PIPELINE = "PipelineFrio"       # marca en el registro las versiones que van de df_total a la predicción
DIAS_HISTORIA = 15                 # lag7 y media móvil de 14 días + el día a predecir


# =============================================================================
# 1. Pipeline del modelo (scaler -> selección -> modelo). Lo usa también notebooks/modelado.ipynb
# =============================================================================

_TRANSFORMADORES = {
    "yeo-johnson": lambda: PowerTransformer(method="yeo-johnson", standardize=False),
    # n_quantiles < filas del fold más chico (~420 con 5 folds)
    "quantile": lambda: QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEMILLA),
}
_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}

# clima y mes (COLS_SIN_LIMITES de preprocess) + producción, con sus lags y medias móviles
PREFIJOS_EXTERNOS = (*COLS_SIN_LIMITES, "hl cerveza envasada")


def columnas_externas(columnas) -> list[str]:
    """Variables externas que siempre entran al modelo. Las interacciones (" x ") y los ratios (" / ")
    no cuentan: mezclan con variables internas (p. ej. `temp max (°c) x frio (kw)`)."""
    return [c for c in columnas if str(c).startswith(PREFIJOS_EXTERNOS) and " x " not in c and " / " not in c]


def construir_pipeline(estimador, columnas, cfg: dict, rfe_estimador=None, memory=None) -> Pipeline:
    """transformador -> scaler (menos mes_sin/mes_cos) -> selección (solo internas; las externas pasan
    directo) -> modelo. `memory`: caché de joblib para la búsqueda (en producción, None)."""
    num = [c for c in columnas if c not in COLS_CICLICAS]
    pasos_num = []
    if cfg["transformador"] != "ninguno":
        pasos_num.append(("trans", _TRANSFORMADORES[cfg["transformador"]]()))
    if cfg["scaler"] != "ninguno":
        pasos_num.append(("scaler", _SCALERS[cfg["scaler"]]()))
    prep = ColumnTransformer(
        [("num", Pipeline(pasos_num) if pasos_num else "passthrough", num)],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    sel = cfg["seleccion"]
    filtro = (SelectorFiltro(cfg["filtro_metodo"], cfg["k"], cfg["umbral_redundancia"])
              if sel in ("filtro", "filtro+rfe") else "passthrough")
    if sel == "filtro+rfe":
        wrapper = RFE(rfe_estimador, n_features_to_select=cfg["n_rfe"], step=0.1)
    elif sel == "lasso":
        wrapper = SelectFromModel(Lasso(alpha=cfg["alpha_sel"], max_iter=20_000, random_state=SEMILLA))
    else:
        wrapper = "passthrough"

    externas = columnas_externas(columnas)
    internas = [c for c in columnas if c not in externas]
    if sel == "ninguna":
        seleccion = "passthrough"
    else:
        # la selección solo ve las internas; las externas se suman sin tocar (ColumnTransformer le pasa y al filtro/RFE)
        seleccion = ColumnTransformer(
            [("sel", Pipeline([("filtro", filtro), ("wrapper", wrapper)]), internas),
             ("externas", "passthrough", externas)],
            verbose_feature_names_out=False,
        )
    pipe = Pipeline([("prep", prep), ("seleccion", seleccion), ("modelo", estimador)], memory=memory)
    return pipe.set_output(transform="pandas")


def columnas_elegidas(pipe: Pipeline) -> list[str]:
    """Columnas que llegan al modelo (después de todos los pasos de selección)."""
    return list(pipe[:-1].get_feature_names_out())


def crear_mlp(**params) -> TransformedTargetRegressor:
    """Red neuronal (MLP) con el target estandarizado.

    El target está en kW (~10000) y los pesos de la red arrancan cerca de 0: sin estandarizarlo
    la red entrena muy mal. El escalado de y se ajusta solo con los datos de entrenamiento que
    recibe .fit() y viaja dentro del objeto, así que .predict() devuelve kW directamente."""
    return TransformedTargetRegressor(regressor=MLPRegressor(**params), transformer=StandardScaler())


ESTIMADORES = {"mlp": crear_mlp, "lasso": Lasso}   # los modelos que hay en el registro


def crear_modelo(nombre: str, params: dict, cfg: dict, columnas) -> Pipeline:
    """Pipeline sin entrenar. El RFE usa el mismo modelo, salvo el MLP (sin coef_) que usa Ridge."""
    clase = ESTIMADORES[nombre]
    rfe_est = Ridge() if nombre == "mlp" else clase(**params)
    return construir_pipeline(clase(**params), columnas, cfg, rfe_est)


@contextmanager
def _sin_avisos():
    """Lasso/MLP avisan "no convergió" y Yeo-Johnson desborda en columnas enormes (igual que en el notebook)."""
    with warnings.catch_warnings():
        for categoria in (ConvergenceWarning, RuntimeWarning, UserWarning):
            warnings.simplefilter("ignore", categoria)
        yield


# =============================================================================
# 2. Configuración desde los registros
# =============================================================================

def normalizar_version(version: str) -> str:
    """"1.0.0" o "v1.0.0" -> "v1.0.0"; cualquier otra cosa es un error."""
    v = str(version).strip()
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", v):
        raise ValueError(f"Versión inválida {version!r}: tiene que ser vMAJOR.MINOR.PATCH (p. ej. v1.0.0)")
    return v if v.startswith("v") else f"v{v}"


def cargar_config(modelo: str = MODELO, version: str | None = None,
                  registro: str | Path = RUTA_REGISTRO_MODELOS, lineage: str | Path = RUTA_LINEAGE) -> dict:
    """{"registro": entrada del modelo, "linaje": entrada del dataset con el que se entrenó}.
    version=None -> la versión de ese modelo con menor MAE de validación."""
    entradas = [e for e in cargar_registro(registro) if e["modelo"] == modelo]
    if version is not None:
        version = normalizar_version(version)
        entradas = [e for e in entradas if e["version"] == version]
    else:
        entradas = sorted((e for e in entradas if e.get("metricas", {}).get("mae_val") is not None),
                          key=lambda e: e["metricas"]["mae_val"], reverse=True)
    if not entradas:
        raise KeyError(f"No hay {modelo} {version or 'con MAE de validación'} en {registro}")
    entrada = entradas[-1]

    sha = entrada["dataset"]["sha256"]
    linajes = [e for e in cargar_registro(lineage) if e["output"]["sha256"] == sha]
    if not linajes:
        raise KeyError(f"El dataset {sha[:12]}... de {modelo} {entrada['version']} no está en {lineage}")
    return {"registro": entrada, "linaje": linajes[-1]}


def _paso_linaje(linaje: dict, paso: str) -> dict:
    return next(t["config"] for t in linaje["transformaciones"] if t["paso"] == paso)


def params_modelo(entrada: dict) -> dict:
    """Hiperparámetros + arquitectura (el JSON guarda las tuplas como listas)."""
    params = {**entrada["hiperparametros"], **entrada["arquitectura"]}
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return params


def cfg_pipeline(entrada: dict) -> dict:
    """Configuración de construir_pipeline (sin la lista informativa de externas)."""
    return {k: v for k, v in entrada["preprocesamiento"].items() if k != "externas_siempre"}


def crear_derivadas(linaje: dict) -> ConfigDerivadas:
    cfg = dict(_paso_linaje(linaje, "agregar_derivadas"))
    for clave in ("ratios", "interacciones"):
        cfg[clave] = [tuple(p) for p in cfg[clave]]
    return ConfigDerivadas(**cfg)


def crear_preprocess(linaje: dict) -> Preprocess:
    """Preprocess sin entrenar con la misma configuración con la que se generó el dataset (ver src/main.py)."""
    cfg = _paso_linaje(linaje, "Preprocess")
    outliers = cfg["outliers"] and {**cfg["outliers"], "percentiles": tuple(cfg["outliers"]["percentiles"])}
    return Preprocess(
        invalidos=cfg["invalidos"] and ConfigInvalidos(**cfg["invalidos"]),
        outliers=outliers and ConfigOutliers(**outliers),
        faltantes=cfg["faltantes"] and ConfigFaltantes(**cfg["faltantes"]),
        transformer=None,   # transformador y scaler van dentro del pipeline del modelo
        scaler=None,
        split=ConfigSplit(**_paso_linaje(linaje, "split temporal")),
    )


# =============================================================================
# 3. Pipeline completo: dato limpio -> predicción del frío de mañana
# =============================================================================

class PipelineFrio(BaseEstimator):
    """agregar_derivadas -> Preprocess -> pipeline del modelo.

    Entrada: días consecutivos como los de data/clean/df_total.xlsx (columna `fecha` + variables de planta,
    clima y mes). La salida de cada fila es el frío (kW) predicho para el DÍA SIGUIENTE. Las variables
    derivadas miran hasta 14 días hacia atrás: para predecir hay que pasar al menos DIAS_HISTORIA días.
    """

    def __init__(self, modelo: str = MODELO, params_modelo: dict | None = None, cfg_pipeline: dict | None = None,
                 cfg_derivadas: ConfigDerivadas | None = None, preprocess: Preprocess | None = None):
        self.modelo = modelo
        self.params_modelo = params_modelo
        self.cfg_pipeline = cfg_pipeline
        self.cfg_derivadas = cfg_derivadas
        self.preprocess = preprocess

    def _derivadas(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_FECHA not in df.columns:
            raise KeyError(f"Falta la columna {COL_FECHA!r}")
        df = df.assign(**{COL_FECHA: pd.to_datetime(df[COL_FECHA])}).sort_values(COL_FECHA)
        return agregar_derivadas(df, self.cfg_derivadas)   # sin descartar filas: rompería los lags

    def _features(self, d: pd.DataFrame) -> pd.DataFrame:
        """Variables tratadas, indexadas por fecha. Sin el target: si no, Preprocess tira las filas sin él."""
        X = self.preprocess_.transform(d.drop(columns=TARGET, errors="ignore"))
        return X.set_index(COL_FECHA)

    def ajustar_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta solo Preprocess (límites de outliers, imputador) con `df`. Devuelve df con las derivadas."""
        d = self._derivadas(df)
        self.preprocess_ = copy.deepcopy(self.preprocess).fit(d)
        return d

    def transformar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Días limpios -> variables listas para el modelo (derivadas + Preprocess), indexadas por fecha."""
        return self._features(self._derivadas(df))

    def columnas_entrada(self) -> list[str]:
        """Columnas (sin las derivadas) con las que se ajustó Preprocess: las que conoce de un df de entrada."""
        return [c for c in self.preprocess_._features_fit if c not in self._origen_derivadas()]

    def _origen_derivadas(self) -> dict[str, set[str]]:
        """{variable derivada: columnas de las que sale}, con los mismos nombres que arma agregar_derivadas."""
        cfg, cols = self.cfg_derivadas, self.preprocess_._features_fit
        origen = {f"{c} lag{k}": {c} for c in cfg.cols_temporales for k in cfg.lags or []}
        origen |= {f"{c} mm{v}": {c} for c in cfg.cols_temporales for v in cfg.ventanas or []}
        origen |= {f"{a} / {b}": {a, b} for a, b in cfg.ratios or []}
        origen |= {f"{a} x {b}": {a, b} for a, b in cfg.interacciones or []}
        if cfg.meta:   # igual que agregar_cumplimiento_meta
            reales = {_clave(c): c for c in cols if c.endswith("hl") and not c.startswith("meta")}
            for m in (c for c in cols if c.startswith("meta")):
                if real := reales.get(_clave(m.removeprefix("meta "))):
                    origen[f"cumpl {real.removesuffix('/hl').removesuffix('/ hl').strip()}"] = {real, m}
        return origen

    def columnas_necesarias(self, columnas_modelo) -> list[str]:
        """Columnas de planta sin las que no se puede predecir: las que pide ConfigDerivadas y las que usa
        el modelo (directamente o a través de una derivada). Clima y mes no cuentan: no vienen en el Excel."""
        cfg, origen = self.cfg_derivadas, self._origen_derivadas()
        necesarias = {*cfg.cols_temporales, *(c for par in [*cfg.ratios, *cfg.interacciones] for c in par)}
        for c in columnas_modelo:
            necesarias |= origen.get(c, {c})
        return sorted(c for c in necesarias if c not in COLS_SIN_LIMITES)

    def fit(self, df: pd.DataFrame, y=None) -> "PipelineFrio":
        """df: días limpios con la columna target (frío de mañana). Las filas sin target no entrenan
        el modelo, pero sí cuentan para los lags y para ajustar Preprocess (igual que en src/main.py)."""
        d = self.ajustar_preprocess(df)
        X = self._features(d)
        y = d.set_index(COL_FECHA)[TARGET]
        ok = y.notna()
        with _sin_avisos():
            self.modelo_ = crear_modelo(self.modelo, self.params_modelo, self.cfg_pipeline, X.columns)
            self.modelo_.fit(X[ok], y[ok])
        self.columnas_ = columnas_elegidas(self.modelo_)
        self.entrenado_hasta_ = d[COL_FECHA].max()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predicción para cada fila (índice = fecha del dato; el valor es el frío del día siguiente)."""
        if len(df) < DIAS_HISTORIA:
            warnings.warn(f"Con {len(df)} días los lags y medias móviles salen incompletos "
                          f"(se imputan); conviene pasar al menos {DIAS_HISTORIA}.")
        X = self._features(self._derivadas(df))
        with _sin_avisos():
            return pd.Series(self.modelo_.predict(X), index=X.index, name=f"pred {TARGET}")

    def predecir_manana(self, df: pd.DataFrame) -> pd.Series:
        """Frío predicho para el día siguiente al último de `df` (Serie de un valor, índice = ese día)."""
        pred = self.predict(df).iloc[[-1]]
        pred.index = pred.index + timedelta(days=1)
        return pred.rename_axis("fecha_predicha")


# =============================================================================
# 4. Entrenamiento
# =============================================================================

def _metricas(y: pd.Series, pred: pd.Series) -> dict:
    ok = y.notna()
    y, pred = y[ok], pred.loc[y.index]
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def _entorno() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__}

def _pipeline_y_datos(cfg: dict) -> tuple[PipelineFrio, pd.DataFrame, pd.DataFrame, str]:
    """(PipelineFrio sin entrenar, train, val, sha256 de la entrada) para una config de cargar_config()."""
    entrada, linaje = cfg["registro"], cfg["linaje"]
    ruta_in = ROOT / linaje["input"]["archivo"]
    if not ruta_in.exists():
        raise FileNotFoundError(f"Falta {linaje['input']['archivo']}: es el dataset limpio con el que se ajusta "
                                f"el preprocesamiento de {entrada['modelo']} {entrada['version']}")
    sha_in = huella(ruta_in)
    if sha_in != linaje["input"]["sha256"]:
        print(f"AVISO: {linaje['input']['archivo']} cambió desde que se generó el dataset del modelo "
              f"({linaje['input']['sha256'][:12]}... -> {sha_in[:12]}...): el split y los datos no son los mismos.")
    preprocess = crear_preprocess(linaje)
    train, val, _ = preprocess.train_test_split(cargar_dataset_limpio(ruta_in))   # test no se usa
    pipe = PipelineFrio(entrada["modelo"], params_modelo(entrada), cfg_pipeline(entrada),
                        crear_derivadas(linaje), preprocess)
    return pipe, train, val, sha_in


def preprocesado_registrado(modelo: str, version: str, registro: str | Path = RUTA_REGISTRO_MODELOS) -> PipelineFrio:
    """PipelineFrio con solo el preprocesamiento ajustado (sin `modelo_`), para las versiones del registro que
    son un Pipeline sobre dataset_final.csv: derivadas + Preprocess ajustado con train, como se generó ese dataset."""
    pipe, train, _, _ = _pipeline_y_datos(cargar_config(modelo, version, registro))
    pipe.ajustar_preprocess(train)
    return pipe


def _version_libre(modelo: str, version: str, registro: str | Path) -> str:
    version = normalizar_version(version)
    if any(e["modelo"] == modelo and e["version"] == version for e in cargar_registro(registro)):
        raise ValueError(f"{modelo} {version} ya está en {Path(registro).name}: elegí otra versión")
    return version


def entrenar(modelo: str = MODELO, version: str | None = None, base: str | None = None, datos: str = "train+val",
             carpeta: str | Path = RUTA_MODELOS, registro: str | Path = RUTA_REGISTRO_MODELOS,
             guardar: bool = True) -> tuple[PipelineFrio, dict]:
    """Entrena PipelineFrio con la config de la versión `base` del registro.

    datos="train+val": modelo para producción (no queda nada para medirlo; test sigue reservado). Se guarda en
        models/{modelo}_{version}.joblib y se agrega como una entrada nueva `version` en model_registry.json.
    datos="train": verificación; reproduce `base` y lo mide en val contra el registro. No guarda nada.
    base=None -> la versión de `modelo` con menor MAE de validación.
    """
    if datos not in ("train", "train+val"):
        raise ValueError(f"datos tiene que ser 'train' o 'train+val', no {datos!r}")
    guardar = guardar and datos == "train+val"
    if guardar:
        if version is None:
            raise ValueError("Falta la versión con la que se registra el modelo (p. ej. version='v1.0.0')")
        version = _version_libre(modelo, version, registro)   # antes de entrenar: falla rápido
    cfg = cargar_config(modelo, base, registro)
    entrada, linaje = cfg["registro"], cfg["linaje"]
    base = entrada["version"]
    print(f"Config de {modelo} {base} ({entrada['archivo']})")

    pipe, train, val, sha_in = _pipeline_y_datos(cfg)
    datos_fit = train if datos == "train" else pd.concat([train, val])
    t0 = datetime.now()
    pipe.fit(datos_fit)
    seg = (datetime.now() - t0).total_seconds()
    print(f"{modelo} (config {base}) entrenado con {datos} ({len(datos_fit)} días, "
          f"{datos_fit[COL_FECHA].min():%Y-%m-%d} a {datos_fit[COL_FECHA].max():%Y-%m-%d}) en {seg:.1f} s | "
          f"{len(pipe.columnas_)} variables")

    y_fit = datos_fit.set_index(COL_FECHA)[TARGET]
    metricas = {f"{k}_train": v for k, v in _metricas(y_fit, pipe.predict(datos_fit)).items()}
    if datos == "train":
        # val necesita la historia de train para sus lags
        pred_val = pipe.predict(pd.concat([train, val])).loc[pd.to_datetime(val[COL_FECHA])]
        metricas |= {f"{k}_val": v for k, v in _metricas(val.set_index(COL_FECHA)[TARGET], pred_val).items()}
        reg = entrada["metricas"]["mae_val"]
        print(f"MAE val {metricas['mae_val']:.2f} | registro {reg:.2f} | diferencia {metricas['mae_val'] - reg:+.2f}")
        if pipe.columnas_ != entrada["columnas"]:
            print("AVISO: las variables elegidas no son las del registro")
    print("Métricas:", {k: round(v, 3) for k, v in metricas.items()})

    # misma forma que las entradas que registra notebooks/modelado.ipynb, más `tipo`
    nueva = {
        "modelo": modelo,
        "version": version,
        "tipo": TIPO_PIPELINE,
        "archivo": None,
        "sha256": None,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "version_previa": base,
        "estudio_optuna": entrada.get("estudio_optuna"),
        "corrida": entrada.get("corrida"),
        # dataset: el del linaje de la base, así cargar_config() también funciona con esta versión
        **{k: entrada[k] for k in ("dataset", "arquitectura", "hiperparametros", "preprocesamiento")},
        "metricas": metricas,   # con train+val solo hay métricas de train: val está adentro y test reservado
        "entrenado_con": {"split": datos, "n_filas": len(datos_fit),
                          "desde": f"{datos_fit[COL_FECHA].min():%Y-%m-%d}",
                          "hasta": f"{datos_fit[COL_FECHA].max():%Y-%m-%d}",
                          "entrada": {"archivo": linaje["input"]["archivo"], "sha256": sha_in},
                          "seg": round(seg, 1)},
        "n_features": len(pipe.columnas_),
        "columnas": pipe.columnas_,
        "semilla": entrada.get("semilla"),
        "entorno": _entorno(),
        "cargar_con": None,
    }
    if guardar:
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / f"{modelo}_{version}.joblib"
        joblib.dump(pipe, ruta, compress=3)
        archivo = ruta.relative_to(ROOT).as_posix() if ruta.is_relative_to(ROOT) else str(ruta)
        nueva |= {"archivo": archivo, "sha256": huella(ruta),
                  "cargar_con": f"sys.path.insert(0, 'src'); joblib.load('{archivo}').predecir_manana(df)"}
        guardar_registro(cargar_registro(registro) + [nueva], registro)
        print(f"Guardado: {archivo} ({nueva['sha256'][:12]}...)")
        print(f"Registrado en {Path(registro).name}: {modelo} {version} (config de {base})")
    return pipe, nueva


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el pipeline completo (derivadas + Preprocess + modelo).")
    parser.add_argument("--modelo", default=MODELO, choices=list(ESTIMADORES))
    parser.add_argument("--version", help="versión con la que se registra el modelo nuevo (p. ej. v1.0.0); "
                                          "si falta, se pide por teclado")
    parser.add_argument("--base", help="versión del registro de la que se toma la config "
                                       "(por defecto, la de menor MAE de validación)")
    parser.add_argument("--datos", default="train+val", choices=["train+val", "train"],
                        help="train = solo verificar la base contra val, sin guardar")
    args = parser.parse_args()
    version = args.version
    if args.datos == "train+val" and not version:
        version = input("Versión del modelo nuevo (p. ej. v1.0.0): ")
    entrenar(args.modelo, version, args.base, args.datos)


if __name__ == "__main__":
    # Se importa a sí mismo: así las clases quedan como `train_model.PipelineFrio` (y no `__main__.PipelineFrio`)
    # y el .joblib se puede cargar desde cualquier script o notebook.
    import train_model
    train_model.main()
