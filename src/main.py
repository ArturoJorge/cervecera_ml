from preprocess import (
    RUTA_LIMPIO, cargar_dataset_limpio, ConfigDerivadas, agregar_derivadas, Preprocess,
    ConfigInvalidos, ConfigOutliers, ConfigFaltantes, ConfigSplit, guardar_dataset_final,
)


def main():
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


if __name__ == "__main__":
    main()
