import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import RUTA_FINAL, TARGET, cargar_registro, guardar_registro, huella
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from preprocess import (
    RUTA_LIMPIO, cargar_dataset_limpio, ConfigDerivadas, agregar_derivadas, Preprocess,
    ConfigInvalidos, ConfigOutliers, ConfigFaltantes, ConfigSplit, guardar_dataset_final, construir_dataset
)



def _metricas(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    ok = ~np.isnan(pred)  # los primeros días de un rolling quedan en NaN
    y, pred = y[ok], pred[ok]
    return {
    "mae": float(mean_absolute_error(y, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    "r2": float(r2_score(y, pred)),
    }

def cargar_datos(ruta: str | Path = RUTA_FINAL, incluir_test: bool = False) -> pd.DataFrame:
    """Lee dataset_final.csv indexado por fecha. Por defecto descarta las filas de test:
    ni siquiera quedan en memoria. Pasar incluir_test=True solo para la evaluación final."""
    df = pd.read_csv(ruta, parse_dates=["fecha"]).sort_values("fecha").set_index("fecha")
    if not incluir_test:
        df = df[df["split"] != "test"]
    return df

def main():
    # Crudos -> data/clean/df_total.xlsx (limpieza + clima + mes cíclico + target)
    df_total = construir_dataset(guardar=RUTA_LIMPIO)

    """"
    df = cargar_dataset_limpio(RUTA_LIMPIO)
    print(df.shape)

    cfg_der = ConfigDerivadas()
    df = agregar_derivadas(df, cfg_der, verbose=True)

    pp = Preprocess(
        invalidos=ConfigInvalidos(accion="nan"),
        outliers=ConfigOutliers(metodo="iqr", accion="clip"),   # clip: val/test no pierden valores legítimos
        transformer=None,   # van dentro del Pipeline del modelo (el CSV queda sin escalar)
        scaler=None,
        faltantes=ConfigFaltantes(estrategia="knn", n_vecinos=5),
        split=ConfigSplit(test_size=0.15, val_size=0.15),
        verbose=True,
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = pp.run(df)
    print(pp.resumen())

    # La selección de variables ya no se hace acá: vive dentro del Pipeline del modelo (ver modelado).
    guardar_dataset_final(
        pp, {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)},
        script="src/main.py", cfg_derivadas=cfg_der,    
    )
    """

    
    


if __name__ == "__main__":
    main()

