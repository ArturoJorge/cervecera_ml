
from preprocess import *

cleanpath = Path("data/clean/df_total.xlsx")

df = construir_dataset(guardar=cleanpath)

pp = Preprocess(
        invalidos=ConfigInvalidos(accion="nan"),
        outliers=ConfigOutliers(metodo="zscore_mod", accion="clip"),
        transformer=ConfigTransformer(metodo="yeo-johnson"),
        scaler=ConfigScaler(metodo="standard"),
        faltantes=ConfigFaltantes(estrategia="knn", n_vecinos=5),  # mean | median | interpolate | knn
        split=ConfigSplit(test_size=0.15),
    )
X_train, X_test, y_train, y_test = pp.run(df)
print(pp.resumen())



