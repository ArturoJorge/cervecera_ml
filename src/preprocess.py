"""Preprocesamiento del dataset de la cervecera.

Contiene:
1. Construcción del dataset limpio a partir de los .xlsx crudos (mismos pasos que Eda),
   o carga directa del dataset ya limpio.
2. Versionado del dataset en data/logs/data_log.json.
3. La clase `Preprocess`, configurable, para: train/test split, datos inválidos,
   valores faltantes y outliers.

Ejemplo de uso
--------------
    from src.preprocess import *

    df = cargar_dataset_limpio()
    # o reconstruirlo desde los crudos:
    # df = construir_dataset(crit_kpi="mean", guardar=RUTA_LIMPIO, versionar=True,
    #                        transformaciones=["..."], comentario="...")

    pp = Preprocess(
        invalidos=ConfigInvalidos(accion="nan"),
        outliers=ConfigOutliers(metodo="zscore_mod", accion="clip"),
        faltantes=ConfigFaltantes(estrategia="knn", n_vecinos=5),  # mean | median | interpolate | knn
        split=ConfigSplit(test_size=0.15),
    )
    X_train, X_test, y_train, y_test = pp.run(df)
    print(pp.resumen())

Para procesar un solo .xlsx:
    df_uno = procesar_excel("data/raw/Totalizadores Planta 2020_2022.xlsx")
    df = juntar_datasets([df_uno])   # mismo formato que con varios archivos
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import reduce
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Rutas y constantes
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RUTA_LIMPIO = ROOT / "data" / "clean" / "df_total.xlsx"
RUTA_REGISTRO = ROOT / "data" / "logs" / "data_log.json"

RAW_DEFAULT = [
    RAW_DIR / "Totalizadores Planta - 2021_2023.xlsx",
    RAW_DIR / "Totalizadores Planta 2020_2022.xlsx",
    RAW_DIR / "Totalizadores Planta 2022_2023.xlsx",
]

COL_FECHA = "fecha"
TARGET = "frio_mañana (kw)"
HOJA_KPI = "Consolidado KPI"
CRITERIOS_KPI = {"sum", "mean", "median"}
RANDOM_STATE = 42

# =============================================================================
# 1. Carga / construcción del dataset
# =============================================================================

def cargar_dataset_limpio(ruta: str | Path = RUTA_LIMPIO) -> pd.DataFrame:
    """Lee el dataset ya limpio (data/clean/df_total.xlsx)."""
    df = pd.read_excel(ruta)
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA])
    return df


def _parsear_dia_hora(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Devuelve (día normalizado, hora como timedelta) a partir de las columnas 'dia' y 'hora'."""
    dia = pd.to_datetime(df["dia"], errors="coerce").dt.normalize()
    h = df["hora"]
    if pd.api.types.is_datetime64_any_dtype(h):
        h = h.dt.strftime("%H:%M:%S")
    hora = pd.to_timedelta(h.astype(str), errors="coerce")
    return dia, hora


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, se queda solo con los cierres de día (>= 23:30) y crea `fecha`.
    Agrega `_n` (orden de aparición de las fecha repetida)."""
    df = _normalizar_columnas(df)
    dia, hora = _parsear_dia_hora(df)

    es_cierre = hora >= pd.Timedelta(hours=23, minutes=30)
    df = df[es_cierre].copy()

    df[COL_FECHA] = dia[es_cierre]
    df = df.drop(columns=["dia", "hora"])
    df = df.dropna(subset=[COL_FECHA])
    df["_n"] = df.groupby(COL_FECHA).cumcount()
    return df


def juntar_kpi(hojas: dict[str, pd.DataFrame], crit: str = "mean") -> dict[str, pd.DataFrame]:
    """Agrega todos los registros del día de la hoja 'Consolidado KPI' (sum, mean o median)
    y escribe el resultado en la fila de cierre del día. Devuelve una copia del dict de hojas."""
    if crit not in CRITERIOS_KPI:
        raise ValueError(f"crit debe ser uno de {sorted(CRITERIOS_KPI)}, no {crit!r}")

    hojas = dict(hojas)
    if HOJA_KPI not in hojas:
        return hojas

    df_t = _normalizar_columnas(hojas[HOJA_KPI])
    dia, hora = _parsear_dia_hora(df_t)
    es_cierre = (hora >= pd.Timedelta(hours=23, minutes=30)) & (hora < pd.Timedelta(days=1))

    for col in df_t.columns.drop(["dia", "hora"]):
        if not pd.api.types.is_numeric_dtype(df_t[col]):
            continue
        agrupado = df_t[col].groupby(dia)
        if crit == "sum":
            valor_dia = agrupado.transform(lambda s: s.sum(min_count=1))
        else:
            valor_dia = agrupado.transform(crit)
        df_t.loc[es_cierre, col] = valor_dia[es_cierre]

    hojas[HOJA_KPI] = df_t
    return hojas


def elegir_mas_similar(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada fecha repetida se queda con la fila con menor desviación respecto
    al día anterior y al siguiente. Quita `_n`."""
    df = df.sort_values([COL_FECHA, "_n"]).reset_index(drop=True)
    num = df.select_dtypes("number").columns.drop(["_n", "id"], errors="ignore")
    escala = df[num].std().replace(0, np.nan)

    por_dia = df.groupby(COL_FECHA)[num].median()
    anterior = por_dia.shift(1).loc[df[COL_FECHA]].set_axis(df.index)
    siguiente = por_dia.shift(-1).loc[df[COL_FECHA]].set_axis(df.index)

    desv_ant = ((df[num] - anterior).abs() / escala).mean(axis=1, skipna=True)
    desv_sig = ((df[num] - siguiente).abs() / escala).mean(axis=1, skipna=True)
    desv = pd.concat([desv_ant, desv_sig], axis=1).mean(axis=1, skipna=True).fillna(np.inf)

    elegidas = desv.groupby(df[COL_FECHA]).idxmin()
    return df.loc[elegidas].drop(columns="_n").reset_index(drop=True)


def unir_hojas(hojas: dict[str, pd.DataFrame], verbose: bool = False) -> pd.DataFrame:
    """Limpia cada hoja, colapsa fechas repetidas y une todas las hojas por `fecha`."""
    partes = []
    for nombre, df in hojas.items():
        df = limpiar(df)
        n_antes = len(df)
        df = elegir_mas_similar(df).set_index(COL_FECHA)
        if verbose and len(df) < n_antes:
            print(f"'{nombre}': {n_antes - len(df)} filas colapsadas por fecha repetida")
        partes.append(df)
    return reduce(lambda a, b: a.combine_first(b), partes)


def procesar_excel(ruta: str | Path, crit_kpi: str = "mean", verbose: bool = False) -> pd.DataFrame:
    """Procesa UN .xlsx crudo (todas sus hojas). Devuelve un df indexado por `fecha`."""
    hojas = pd.read_excel(ruta, sheet_name=None)
    hojas = juntar_kpi(hojas, crit=crit_kpi)
    return unir_hojas(hojas, verbose=verbose)


def juntar_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Une varios resultados de `procesar_excel`. 
    devuelve un df ordenado con `fecha` como columna."""
    if not dfs:
        raise ValueError("No se pasó ningún dataset para juntar")
    return reduce(lambda a, b: a.combine_first(b), dfs).copy().sort_index().reset_index()


def borrar_columnas_vacias(
    df: pd.DataFrame, min_nulos: int = 1100, umbral_varianza: float = 10
) -> tuple[pd.DataFrame, list[str]]:
    """Elimina columnas con >= min_nulos nulos o con varianza <= umbral_varianza.
    Devuelve (df sin esas columnas, lista de columnas eliminadas)."""
    eliminar = []
    for col in df.columns:
        if col == COL_FECHA:
            continue
        nulos = df[col].isnull().sum()
        varianza = df[col].var() if pd.api.types.is_numeric_dtype(df[col]) else None
        if nulos >= min_nulos or (varianza is not None and varianza <= umbral_varianza):
            eliminar.append(col)
    return df.drop(columns=eliminar), eliminar


def convertir_object_a_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de texto a numéricas (Int64 si son enteras, float64 si no)."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        numericos = pd.to_numeric(df[col], errors="coerce")
        validos = numericos.dropna()
        if (validos % 1 == 0).all():
            df[col] = numericos.astype("Int64")
        else:
            df[col] = numericos.astype("float64")
    return df


def agregar_target(df: pd.DataFrame, col_origen: str = "frio (kw)", nombre: str = TARGET) -> pd.DataFrame:
    """Crea la variable objetivo: el valor de `col_origen` del día siguiente."""
    df = df.copy()
    df[nombre] = df[col_origen].shift(-1)
    return df


def construir_dataset(
    rutas: list[str | Path] = RAW_DEFAULT,
    crit_kpi: str = "mean",
    min_nulos: int = 1100,
    umbral_varianza: float = 10,
    guardar: str | Path | None = None,
    versionar: bool = False,
    transformaciones: list[str] | None = None,
    comentario: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Limpieza de los .xlsx crudos.

    procesar_excel -> juntar_datasets -> borrar_columnas_vacias
    -> convertir_object_a_numerico -> agregar_target.

    Si `guardar` es una ruta, escribe el xlsx. Si además `versionar=True`, lo registra
    en data_log.json (requiere `guardar`).
    """
    partes = [procesar_excel(r, crit_kpi=crit_kpi, verbose=verbose) for r in rutas]
    df = juntar_datasets(partes)
    df, eliminadas = borrar_columnas_vacias(df, min_nulos=min_nulos, umbral_varianza=umbral_varianza)
    df = convertir_object_a_numerico(df)
    df = agregar_target(df)

    if verbose:
        print(f"Dataset construido: {df.shape} | columnas eliminadas: {len(eliminadas)}")

    if guardar is not None:
        Path(guardar).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(guardar, index=False)
    if versionar:
        if guardar is None:
            raise ValueError("Para versionar hay que pasar `guardar` (se versiona el archivo escrito)")
        trans = transformaciones or [f"construir_dataset con crit_kpi={crit_kpi!r}"]
        registrar_version(guardar, trans, comentario=comentario)
    return df


# =============================================================================
# 2. Versionado
# =============================================================================

def huella(ruta: str | Path, algoritmo: str = "sha256", bloque: int = 1 << 20) -> str:
    """Hash hexadecimal del archivo, leído de a 1 MiB."""
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def cargar_registro(registro: str | Path = RUTA_REGISTRO) -> list:
    registro = Path(registro)
    if registro.exists():
        with open(registro, encoding="utf-8") as f:
            return json.load(f)
    return []


def guardar_registro(entradas: list, registro: str | Path = RUTA_REGISTRO) -> None:
    registro = Path(registro)
    registro.parent.mkdir(parents=True, exist_ok=True)
    with open(registro, "w", encoding="utf-8") as f:
        json.dump(entradas, f, indent=2, ensure_ascii=False)


def registrar_version(
    ruta: str | Path,
    transformaciones: list[str],
    comentario: str | None = None,
    registro: str | Path = RUTA_REGISTRO,
) -> dict | None:
    """Registra una nueva versión del archivo solo si su hash cambió.
    Si `comentario` es None se pide por input()."""
    entradas = cargar_registro(registro)
    nuevo_hash = huella(ruta)

    if entradas and entradas[-1]["sha256"] == nuevo_hash:
        print(f"Sin cambios (hash {nuevo_hash[:12]}...): no se crea nueva versión.")
        return None

    if comentario is None:
        comentario = input("Ingrese un comentario para el json del dataframe: ")

    try:
        salida = Path(ruta).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        salida = str(ruta).replace("\\", "/")

    entrada = {
        "version": f"v{len(entradas) + 1}",
        "created_at": date.today().isoformat(),
        "output": salida,
        "sha256": nuevo_hash,
        "parent_sha256": entradas[-1]["sha256"] if entradas else None,
        "transformations": list(transformaciones),
        "comments": comentario,
    }
    entradas.append(entrada)
    guardar_registro(entradas, registro)
    print(f"Nueva versión registrada: {entrada['version']} ({nuevo_hash[:12]}...)")
    return entrada


# =============================================================================
# 3. Configuración del preprocesamiento
# =============================================================================

@dataclass
class ConfigInvalidos:
    """Valores imposibles. Se evalúan sin aprender nada de los datos."""
    no_negativas: list[str] | Literal["all"] = "all"   # columnas que no pueden ser < 0
    max_abs: float | None = 1_000_000                   # |valor| > max_abs es imposible
    rangos: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)  # (mín, máx) por columna
    accion: Literal["nan", "clip"] = "nan"


@dataclass
class ConfigFaltantes:
    estrategia: Literal["mean", "median", "interpolate", "knn"] = "median"
    n_vecinos: int = 5                          # solo para knn
    max_frac_nulos_col: float | None = None     # descarta columnas (en train) con más nulos que esto


@dataclass
class ConfigOutliers:
    metodo: Literal["iqr", "zscore_mod", "percentil"] = "iqr"
    accion: Literal["clip", "nan"] = "clip"
    k_iqr: float = 1.5
    umbral_zmod: float = 3.5
    percentiles: tuple[float, float] = (0.01, 0.99)
    excluir: list[str] = field(default_factory=list)     # columnas a las que no se les tocan outliers


@dataclass
class ConfigSplit:
    # tener en cuenta que la consigna pide 70% de train, 15% de validación y 15% de test
    test_size: float = 0.15


_ESTRATEGIAS_FALTANTES = {"mean", "median", "interpolate", "knn"}
_ACCIONES = {"nan", "clip"}
_METODOS_OUTLIERS = {"iqr", "zscore_mod", "percentil"}
_DEFAULT = object()  # centinela: distingue "no pasé config" de "None = desactivar paso"


# =============================================================================
# 4. Clase Preprocess
# =============================================================================

class Preprocess:
    """Preprocesamiento configurable.

    `fit` aprende los estadísticos (límites de outliers, imputadores, columnas a descartar)
    solo con train; `transform` los aplica en este orden: inválidos -> outliers -> faltantes.
    Cualquier paso se desactiva pasando None en su config. Las filas sin target se eliminan siempre.
    """

    def __init__(
        self,
        target: str = TARGET,
        col_fecha: str = COL_FECHA,
        # ----------- Estos son configuraciones para los tratamientos
        invalidos: ConfigInvalidos | None = _DEFAULT, # valores invalidos
        faltantes: ConfigFaltantes | None = _DEFAULT, # valores faltantes
        outliers: ConfigOutliers | None = _DEFAULT,   # valores atípicos
        split: ConfigSplit = _DEFAULT,                # configuración del train/test split

        verbose: bool = True,
    ):
        self.target = target
        self.col_fecha = col_fecha
        # sin argumento -> config por defecto; None -> paso desactivado
        self.invalidos = ConfigInvalidos() if invalidos is _DEFAULT else invalidos
        self.faltantes = ConfigFaltantes() if faltantes is _DEFAULT else faltantes
        self.outliers = ConfigOutliers() if outliers is _DEFAULT else outliers
        self.split = ConfigSplit() if split in (_DEFAULT, None) else split
        self.verbose = verbose

        if self.faltantes and self.faltantes.estrategia not in _ESTRATEGIAS_FALTANTES:
            raise ValueError(f"Estrategia de faltantes inválida: {self.faltantes.estrategia!r}")
        if self.outliers and self.outliers.metodo not in _METODOS_OUTLIERS:
            raise ValueError(f"Método de outliers inválido: {self.outliers.metodo!r}")
        for cfg in (self.invalidos, self.outliers):
            if cfg and cfg.accion not in _ACCIONES:
                raise ValueError(f"Acción inválida: {cfg.accion!r}")

        self._fitted = False
        self._log: dict[str, dict] = {}

    # ------------------------------------------------------------------ split
    def train_test_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # tiene que ser por fecha porque sino vamos a predecir cualquier día
        df = df.sort_values(self.col_fecha).reset_index(drop=True)
        corte = int(round(len(df) * (1 - self.split.test_size)))
        return df.iloc[:corte].copy(), df.iloc[corte:].copy()

    # ------------------------------------------------------------------ auxiliares
    def _features(self, df: pd.DataFrame) -> list[str]:
        return [
            c for c in df.select_dtypes("number").columns
            if c not in (self.target, self.col_fecha)
        ]

    def _limites_invalidos(self, columnas: pd.Index) -> tuple[pd.Series, pd.Series]:
        cfg = self.invalidos
        lo = pd.Series(-np.inf, index=columnas)
        hi = pd.Series(np.inf, index=columnas)
        no_neg = columnas if cfg.no_negativas == "all" else columnas.intersection(cfg.no_negativas)
        lo[no_neg] = 0.0
        if cfg.max_abs is not None:
            lo, hi = lo.clip(lower=-cfg.max_abs), hi.clip(upper=cfg.max_abs)
        for c, (r_lo, r_hi) in cfg.rangos.items():
            if c in columnas:
                lo[c] = lo[c] if r_lo is None else r_lo
                hi[c] = hi[c] if r_hi is None else r_hi
        return lo, hi

    def _limites_outliers(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        cfg = self.outliers
        if cfg.metodo == "iqr":
            q1, q3 = X.quantile(0.25), X.quantile(0.75)
            delta = cfg.k_iqr * (q3 - q1)
            return q1 - delta, q3 + delta
        if cfg.metodo == "zscore_mod":
            # |0.6745 (x - mediana) / MAD| > umbral  <=>  |x - mediana| > umbral * MAD / 0.6745 (= MAD "normal")
            mad = stats.median_abs_deviation(X, scale="normal", nan_policy="omit")
            delta = pd.Series(cfg.umbral_zmod * mad, index=X.columns).replace(0, np.inf)  # MAD 0 -> no se toca
            med = X.median()
            return med - delta, med + delta
        p_lo, p_hi = cfg.percentiles
        return X.quantile(p_lo), X.quantile(p_hi)

    @staticmethod
    def _aplicar_limites(X: pd.DataFrame, lo: pd.Series, hi: pd.Series, accion: str) -> tuple[pd.DataFrame, int]:
        """Recorta o pasa a NaN lo que cae fuera de [lo, hi]. Devuelve (X, valores tocados)."""
        lo, hi = lo.fillna(-np.inf), hi.fillna(np.inf)  # columnas sin datos en train: no se tocan
        fuera = X.lt(lo, axis=1) | X.gt(hi, axis=1)
        X = X.clip(lo, hi, axis=1) if accion == "clip" else X.mask(fuera)
        return X, int(fuera.to_numpy().sum())

    def _cols_outliers(self, X: pd.DataFrame) -> pd.Index:
        return X.columns.difference(self.outliers.excluir, sort=False)

    # ------------------------------------------------------------------ fit / transform
    def fit(self, train: pd.DataFrame) -> "Preprocess":
        self._features_fit = self._features(train)
        X = train[self._features_fit].astype("float64")

        if self.invalidos:  # límites fijos, no dependen de los datos
            self._lim_invalidos = self._limites_invalidos(X.columns)
            X, _ = self._aplicar_limites(X, *self._lim_invalidos, "nan")

        if self.outliers:  # límites aprendidos en train (ya sin inválidos)
            cols = self._cols_outliers(X)
            self._lim_outliers = self._limites_outliers(X[cols])
            X[cols], _ = self._aplicar_limites(X[cols], *self._lim_outliers, self.outliers.accion)

        if self.faltantes:
            cfg = self.faltantes
            self._cols_descartar = (
                [] if cfg.max_frac_nulos_col is None
                else X.columns[X.isna().mean() > cfg.max_frac_nulos_col].tolist()
            )
            X = X.drop(columns=self._cols_descartar)
            if cfg.estrategia == "knn":
                # KNN usa distancias: sin escalar, las columnas con valores grandes dominan
                self._escalador = StandardScaler().fit(X)
                self._imputador = KNNImputer(n_neighbors=cfg.n_vecinos, keep_empty_features=True)
                self._imputador.fit(self._escalador.transform(X))
            else:
                # en interpolate, la mediana de train rellena lo que no se pueda interpolar
                estr = "mean" if cfg.estrategia == "mean" else "median"
                self._imputador = SimpleImputer(strategy=estr, keep_empty_features=True).fit(X)

        self._fitted = True  # esto está para que no transformemos sin fittear
        return self

    def _imputar(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.faltantes.estrategia == "knn":
            Z = self._imputador.transform(self._escalador.transform(X))
            valores = self._escalador.inverse_transform(Z)
        else:
            if self.faltantes.estrategia == "interpolate":
                X = X.interpolate(limit_direction="both")  # filas ya ordenadas por fecha
            valores = self._imputador.transform(X)
        return pd.DataFrame(valores, index=X.index, columns=X.columns)

    def transform(self, df: pd.DataFrame, nombre: str = "df") -> pd.DataFrame:
        """Devuelve fecha + target (si están) + features tratadas."""
        if not self._fitted:
            raise RuntimeError("El preprocess no está fitteado (。_。)")
        log = {"filas_inicio": len(df)}
        if self.col_fecha in df.columns:
            df = df.sort_values(self.col_fecha)
        if self.target in df.columns:
            df = df.dropna(subset=[self.target])  # sin target no sirve ni para entrenar ni para evaluar
        X = df[self._features_fit].astype("float64")

        if self.invalidos:
            X, log["invalidos_valores"] = self._aplicar_limites(X, *self._lim_invalidos, self.invalidos.accion)

        if self.outliers:
            cols = self._cols_outliers(X)
            X[cols], log["outliers_valores"] = self._aplicar_limites(X[cols], *self._lim_outliers, self.outliers.accion)

        if self.faltantes:
            X = X.drop(columns=self._cols_descartar)
            log["columnas_descartadas_por_nulos"] = len(self._cols_descartar)
            log["faltantes_valores"] = int(X.isna().to_numpy().sum())
            X = self._imputar(X)

        log["filas_fin"] = len(X)
        self._log[nombre] = log
        if self.verbose:
            print(f"[{nombre}] {log}")
        return df[[c for c in (self.col_fecha, self.target) if c in df.columns]].join(X)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train, nombre="train")

    # ------------------------------------------------------------------ Para probar
    def separar_xy(self, df: pd.DataFrame):
        """X = features numéricas (índice = fecha si existe), y = target."""
        if self.col_fecha in df.columns:
            df = df.set_index(self.col_fecha)
        return df[self._features(df)], df[self.target]

    def run(self, df: pd.DataFrame):
        """Para probar todo junto
        split -> fit en train -> transform en train y test -> (X_train, X_test, y_train, y_test)."""
        train, test = self.train_test_split(df)
        X_train, y_train = self.separar_xy(self.fit_transform(train))
        X_test, y_test = self.separar_xy(self.transform(test, nombre="test"))
        return X_train, X_test, y_train, y_test

    def resumen(self):
        """Configuración usada y qué hizo cada paso en train/test."""
        cfgs = {"invalidos": self.invalidos, "outliers": self.outliers,
                "faltantes": self.faltantes, "split": self.split}
        return {
            "config": {k: asdict(v) if v else None for k, v in cfgs.items()},
            "pasos": self._log,
        }
