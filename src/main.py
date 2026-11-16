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



df=construir_dataset(guardar="data/clean/df_total.xlsx")