"""Selección de variables por filtro, para usar dentro de un Pipeline de scikit-learn.

Vive en un módulo (y no en el notebook) para que los modelos guardados con joblib se puedan
cargar desde cualquier script: `sys.path.insert(0, "src"); joblib.load(...)`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression

RUTA_CACHE = Path(__file__).resolve().parents[1] / "results" / ".cache"
SEMILLA = 42
memoria = joblib.Memory(RUTA_CACHE, verbose=0)


@memoria.cache
def _scores_filtro(X: pd.DataFrame, y: pd.Series, metodo: str) -> tuple[pd.Series, pd.DataFrame]:
    """|score| de cada columna contra y (ordenado de mayor a menor) y |corr| entre columnas
    (sobre rangos si el método es spearman o mutual_info). Cacheado en disco por (X, y, método)."""
    X = X.loc[:, X.nunique() > 1]                      # fuera constantes en este fold
    y = pd.Series(np.asarray(y, dtype=float), index=X.index)
    R = X.rank() if metodo in ("spearman", "mutual_info") else X
    if metodo == "mutual_info":
        s = pd.Series(mutual_info_regression(X, y, random_state=SEMILLA), index=X.columns)
    else:
        Xr, yr = (R, y.rank()) if metodo == "spearman" else (X, y)
        Xc, yc = Xr - Xr.mean(), yr - yr.mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (Xc.mul(yc, axis=0).sum() / np.sqrt((Xc ** 2).sum() * (yc ** 2).sum())).abs()
    s = s.fillna(0).sort_values(ascending=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = R[s.index].corr().abs().fillna(0)
    return s, corr


class SelectorFiltro(BaseEstimator, TransformerMixin):
    """Top-k por score univariado contra y (pearson, spearman o mutual_info), descartando columnas
    redundantes (|corr| >= umbral con una ya elegida). umbral=None o >= 1 -> sin control de redundancia."""

    def __init__(self, metodo="spearman", k=80, umbral=0.95):
        self.metodo, self.k, self.umbral = metodo, k, umbral

    def fit(self, X, y):
        s, corr = _scores_filtro(pd.DataFrame(X), pd.Series(np.asarray(y, dtype=float), index=X.index), self.metodo)
        if self.umbral is None or self.umbral >= 1:
            self.columnas_ = s.index[: self.k].tolist()
            return self
        M = corr.to_numpy()
        elegidas = []
        for i in range(len(s)):
            if all(M[i, j] < self.umbral for j in elegidas):
                elegidas.append(i)
                if len(elegidas) == self.k:
                    break
        self.columnas_ = s.index[elegidas].tolist()
        return self

    def transform(self, X):
        return X[self.columnas_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_, dtype=object)
