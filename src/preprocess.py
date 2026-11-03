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
        faltantes=ConfigFaltantes(estrategia="interpolate"),
        split=ConfigSplit(test_size=0.2, temporal=True),
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
from dataclasses import dataclass, field
from datetime import date
from functools import reduce
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

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
    max_abs: float | None = 1_000_000                   # |valor| >= max_abs es imposible
    rangos: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)  # (mín, máx) por columna
    accion: Literal["nan", "drop", "clip"] = "nan"


@dataclass
class ConfigFaltantes:
    estrategia: Literal["drop", "mean", "median", "ffill", "bfill", "interpolate", "constant"] = "median"
    valor_constante: float = 0
    por_columna: dict[str, str] = field(default_factory=dict)   # override de estrategia por columna
    max_frac_nulos_col: float | None = None                     # descarta columnas (en train) con más nulos que esto
    drop_target_nulo: bool = True


@dataclass
class ConfigOutliers:
    metodo: Literal["iqr", "zscore_mod", "percentil"] | None = "iqr"
    accion: Literal["clip", "nan", "drop"] = "clip"
    k_iqr: float = 1.5
    umbral_zmod: float = 3.5
    percentiles: tuple[float, float] = (0.01, 0.99)
    columnas: list[str] | None = None                    # None = todas las numéricas
    excluir: list[str] = field(default_factory=list)     # el target siempre se excluye


@dataclass
class ConfigSplit:
    # tener en cuenta que la consigna pide 70% de train, 15% de validación y 15% de test
    test_size: float = 0.15



# o((⊙﹏⊙))o. TODO: ENTENDER ESTO
_ESTRATEGIAS_FALTANTES = {"drop", "mean", "median", "ffill", "bfill", "interpolate", "constant"}
_CONSTANTE_ZMOD = 0.6745
_DEFAULT = object()  # centinela: distingue "no pasé config" de "None = desactivar paso"


# =============================================================================
# 4. Clase Preprocess
# =============================================================================

class Preprocess:
    """Preprocesamiento configurable

    `fit` aprende los estadísticos (límites de outliers, medias/medianas, columnas a descartar)
    solo con train; `transform` los aplica en este orden: inválidos -> outliers -> faltantes.
    Cualquier paso se desactiva pasando None en su config.
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

        if self.faltantes:
            for e in [self.faltantes.estrategia, *self.faltantes.por_columna.values()]:
                if e not in _ESTRATEGIAS_FALTANTES:
                    raise ValueError(f"Estrategia de faltantes inválida: {e!r}")

        self._fitted = False
        self._log: dict[str, dict] = {}

    # ------------------------------------------------------------------ split
    def train_test_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.split
        # tiene que ser por fecha porque sino vamos a predecir cualquier día
        df = df.sort_values(self.col_fecha).reset_index(drop=True)
        corte = int(round(len(df) * (1 - cfg.test_size)))
        return df.iloc[:corte].copy(), df.iloc[corte:].copy()
        

    # ------------------------------------------------------------------ auxiliares
    def _features(self, df: pd.DataFrame) -> list[str]:
        return [
            c for c in df.select_dtypes("number").columns
            if c not in (self.target, self.col_fecha)
        ]

    def _a_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pasa las columnas numéricas a float64 (evita problemas con Int64 al imputar o recortar)."""
        df = df.copy()
        num = df.select_dtypes("number").columns
        df[num] = df[num].astype("float64")
        return df

    def _limites_invalidos(self, columnas: list[str]) -> dict[str, tuple[float, float]]:
        cfg = self.invalidos
        no_neg = set(columnas) if cfg.no_negativas == "all" else set(cfg.no_negativas)
        limites = {}
        for c in columnas:
            lo, hi = -np.inf, np.inf
            if c in no_neg:
                lo = 0.0
            if cfg.max_abs is not None:
                lo, hi = max(lo, -cfg.max_abs), min(hi, cfg.max_abs)
            if c in cfg.rangos:
                r_lo, r_hi = cfg.rangos[c]
                lo = lo if r_lo is None else r_lo
                hi = hi if r_hi is None else r_hi
            if np.isfinite(lo) or np.isfinite(hi):
                limites[c] = (lo, hi)
        return limites

    def _limites_outliers(self, s: pd.Series) -> tuple[float, float] | None:
        # TODO: ver si dejamos esto así o lo hacemos con una función porque no necesitamos que sea tan configurable tampoco
        cfg = self.outliers
        s = s.dropna()
        if s.empty:
            return None
        if cfg.metodo == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            return q1 - cfg.k_iqr * iqr, q3 + cfg.k_iqr * iqr
        if cfg.metodo == "zscore_mod":
            med = s.median()
            mad = np.median(np.abs(s - med))
            if mad == 0:
                return None
            delta = cfg.umbral_zmod * mad / _CONSTANTE_ZMOD
            return med - delta, med + delta
        if cfg.metodo == "percentil":
            p_lo, p_hi = cfg.percentiles
            return s.quantile(p_lo), s.quantile(p_hi)
        raise ValueError(f"Método de outliers inválido: {cfg.metodo!r}")

    @staticmethod
    def _aplicar_limites(
        df: pd.DataFrame, limites: dict[str, tuple[float, float]], accion: str
    ) -> tuple[pd.DataFrame, int, int]:
        """Aplica (lo, hi) por columna. Devuelve (df, valores tocados, filas eliminadas)."""
        df = df.copy()
        filas_malas = pd.Series(False, index=df.index)
        tocados = 0
        for c, (lo, hi) in limites.items():
            if c not in df.columns:
                continue
            mascara = (df[c] < lo) | (df[c] > hi)
            tocados += int(mascara.sum())
            if accion == "clip":
                df[c] = df[c].clip(lower=lo, upper=hi)
            elif accion == "nan":
                df.loc[mascara, c] = np.nan
            elif accion == "drop":
                filas_malas |= mascara
            else:
                raise ValueError(f"Acción inválida: {accion!r}")
        eliminadas = int(filas_malas.sum())
        if eliminadas:
            df = df[~filas_malas]
        return df, tocados, eliminadas

    # ------------------------------------------------------------------ fit / transform
    def fit(self, train: pd.DataFrame) -> "Preprocess":
        train = self._a_float(train)
        feats = self._features(train)

        # inválidos: límites fijos, no dependen de los datos
        self._lim_invalidos = self._limites_invalidos(feats) if self.invalidos else {}
        if self.invalidos:
            train, _, _ = self._aplicar_limites(train, self._lim_invalidos, "nan")

        # outliers: límites aprendidos en train (ya sin inválidos)
        self._lim_outliers = {}
        if self.outliers and self.outliers.metodo:
            cols = self.outliers.columnas or feats
            excl = set(self.outliers.excluir) | {self.target, self.col_fecha}
            for c in cols:
                if c in excl or c not in train.columns:
                    continue
                lim = self._limites_outliers(train[c])
                if lim is not None:
                    self._lim_outliers[c] = lim
            if self.outliers.accion != "drop":
                train, _, _ = self._aplicar_limites(train, self._lim_outliers, self.outliers.accion)

        # faltantes: columnas a descartar y valores de imputación
        self._cols_descartar: list[str] = []
        self._valores_imputacion: dict[str, float] = {}
        if self.faltantes:
            cfg = self.faltantes
            if cfg.max_frac_nulos_col is not None:
                frac = train[feats].isna().mean()
                self._cols_descartar = frac[frac > cfg.max_frac_nulos_col].index.tolist()
            restantes = [c for c in feats if c not in self._cols_descartar]
            medianas = train[restantes].median()
            medias = train[restantes].mean()
            for c in restantes:
                estr = cfg.por_columna.get(c, cfg.estrategia)
                if estr == "mean":
                    v = medias[c]
                elif estr == "constant":
                    v = cfg.valor_constante
                else:  # mediana como respaldo para ffill/bfill/interpolate y como estrategia
                    v = medianas[c]
                self._valores_imputacion[c] = 0.0 if pd.isna(v) else float(v)

        self._features_fit = feats
        self._fitted = True # esto está para que no transformemos sin fittear
        return self

    def transform(self, df: pd.DataFrame, nombre: str = "df") -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("El preprocess no está fitteado (。_。)")
        df = self._a_float(df)
        log = {"filas_inicio": len(df)}

        if self.invalidos:
            df, tocados, elim = self._aplicar_limites(df, self._lim_invalidos, self.invalidos.accion)
            log["invalidos_valores"], log["invalidos_filas_eliminadas"] = tocados, elim

        if self.outliers and self.outliers.metodo:
            df, tocados, elim = self._aplicar_limites(df, self._lim_outliers, self.outliers.accion)
            log["outliers_valores"], log["outliers_filas_eliminadas"] = tocados, elim

        if self.faltantes:
            df = self._imputar(df, log)

        log["filas_fin"] = len(df)
        self._log[nombre] = log
        if self.verbose:
            print(f"[{nombre}] {log}")
        return df

    def _imputar(self, df: pd.DataFrame, log: dict) -> pd.DataFrame:
        cfg = self.faltantes
        df = df.drop(columns=[c for c in self._cols_descartar if c in df.columns])
        log["columnas_descartadas_por_nulos"] = len(self._cols_descartar)

        if cfg.drop_target_nulo and self.target in df.columns:
            antes = len(df)
            df = df[df[self.target].notna()]
            log["filas_target_nulo"] = antes - len(df)

        cols = [c for c in self._valores_imputacion if c in df.columns]
        log["faltantes_valores"] = int(df[cols].isna().sum().sum())

        orden = df[self.col_fecha].sort_values().index if self.col_fecha in df.columns else df.index
        filas_drop = []
        for c in cols:
            estr = cfg.por_columna.get(c, cfg.estrategia)
            s = df.loc[orden, c]
            if estr == "drop":
                filas_drop.append(s.isna())
                continue
            if estr == "ffill":
                s = s.ffill()
            elif estr == "bfill":
                s = s.bfill()
            elif estr == "interpolate":
                s = s.interpolate(limit_direction="both")
            # mean / median / constant, y respaldo de lo que quede sin rellenar
            df.loc[orden, c] = s.fillna(self._valores_imputacion[c])

        if filas_drop:
            mascara = pd.concat(filas_drop, axis=1).any(axis=1).reindex(df.index, fill_value=False)
            log["faltantes_filas_eliminadas"] = int(mascara.sum())
            df = df[~mascara]
        return df

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train, nombre="train")

    # ------------------------------------------------------------------ Para probar
    def separar_xy(self, df: pd.DataFrame):
        """X = features numéricas (índice = fecha si existe), y = target."""
        if self.col_fecha in df.columns:
            df = df.set_index(self.col_fecha)
        y = df[self.target]
        X = df[[c for c in self._features(df) if c in df.columns]]
        return X, y

    def run(self, df: pd.DataFrame):
        """Para probar todo junto
        split -> fit en train -> transform en train y test -> (X_train, X_test, y_train, y_test)."""
        train, test = self.train_test_split(df)
        train = self.fit_transform(train)
        test = self.transform(test, nombre="test")
        X_train, y_train = self.separar_xy(train)
        X_test, y_test = self.separar_xy(test)
        X_test = X_test[X_train.columns]
        return X_train, X_test, y_train, y_test

    def resumen(self):
        """Configuración usada y qué hizo cada paso en train/test."""
        from dataclasses import asdict

        return {
            "config": {
                "invalidos": asdict(self.invalidos) if self.invalidos else None,
                "outliers": asdict(self.outliers) if self.outliers else None,
                "faltantes": asdict(self.faltantes) if self.faltantes else None,
                "split": asdict(self.split) if self.split else None,
            },
            "pasos": self._log,
        }
