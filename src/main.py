
from preprocess import *


df = cargar_dataset_limpio(RUTA_LIMPIO)
#df = construir_dataset(guardar=RUTA_LIMPIO, versionar=True, 
#                       comentario="Añadí los meses de forma ciclica para que tengan la distancia apropiada, y añadí la temperatura y la humedad de cada día")
print(df.shape)

# lags, medias móviles, ratios, cumplimiento de meta e interacciones (configurables con ConfigDerivadas)
cfg_der = ConfigDerivadas()
df = agregar_derivadas(df, cfg_der, verbose=True)

pp = Preprocess(
        invalidos=ConfigInvalidos(accion="nan"),
        outliers=ConfigOutliers(metodo="iqr", accion="nan"),
        transformer=ConfigTransformer(metodo="yeo-johnson"),
        scaler=ConfigScaler(metodo="robust"),
        faltantes=ConfigFaltantes(estrategia="knn", n_vecinos=5),  # mean | median | interpolate | knn
        split=ConfigSplit(test_size=0.15, val_size=0.15),  # 70/15/15 temporal
        verbose=True
    )
X_train, X_val, X_test, y_train, y_val, y_test = pp.run(df)
print(pp.resumen())

# selección de variables: config elegida con el estudio de Optuna de preprocesamiento.ipynb (sección 12)
'''
sel = SeleccionVariables(
        filtro=ConfigFiltro(metodo="f_regression", k=60),      # pearson | spearman | f_regression | mutual_info
        wrapper=ConfigWrapper(metodo="lasso", alpha=0.01),     # lasso | rfe | forward | backward
        verbose=True
    )
'''

#X_train = sel.fit_transform(X_train, y_train)
#X_val, X_test = sel.transform(X_val), sel.transform(X_test)

# data/processed/dataset_final.csv + checksum (data/logs/checksums.json) + linaje (data/logs/data_lineage.json)
guardar_dataset_final(pp, {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)},
                      script="src/main.py", cfg_derivadas=cfg_der)



