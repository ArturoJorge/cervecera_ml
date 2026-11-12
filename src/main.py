
from preprocess import *


df = cargar_dataset_limpio(RUTA_LIMPIO)
#df = construir_dataset(guardar=RUTA_LIMPIO, versionar=True, 
#                       comentario="Añadí los meses de forma ciclica para que tengan la distancia apropiada, y añadí la temperatura y la humedad de cada día")
print(df.shape)

pp = Preprocess(
        invalidos=ConfigInvalidos(accion="nan"),
        outliers=ConfigOutliers(metodo="iqr", accion="nan"),
        transformer=ConfigTransformer(metodo="yeo-johnson"),
        scaler=ConfigScaler(metodo="robust"),
        faltantes=ConfigFaltantes(estrategia="knn", n_vecinos=5),  # mean | median | interpolate | knn
        split=ConfigSplit(test_size=0.15),
        verbose=True
    )
X_train, X_test, y_train, y_test = pp.run(df)
print(pp.resumen())



