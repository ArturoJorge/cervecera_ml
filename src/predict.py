import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  
from preprocess import ROOT, cargar_dataset_limpio, huella, construir_dataset



def _clave_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.lstrip("v").split("."))

def load_model(modelo: str="mlp", version: str = "latest"):
    """Carga el modelo y los preprocesadores desde el registro.
    'latest' carga el modelo con la version mas alta.
    """
    entradas = [e for e in registry if e["modelo"] == modelo and PIPELINE in e.get("pipelines", {})]
    if not entradas:
        raise KeyError(f"No hay versiones de {modelo!r} con pipeline {PIPELINE!r} en el registry "
                       f"(correr src/train_model.py)")

    if version == "latest":
        entrada = max(entradas, key=lambda e: _clave_version(e["version"]))
    else:
        entrada = next((e for e in entradas if e["version"] == version), None)
        if entrada is None:
            raise KeyError(f"No existe {modelo} {version} con pipeline {PIPELINE!r} en el registry")

    info = entrada["pipelines"][PIPELINE]
    ruta = ROOT / info["archivo"]


    with open('models/model_registry.json', 'r') as f:
        registry = json.load(f)
    # Logica para encontrar la ultima/mejor version
    model_path = registry[version]['path']
    model = joblib.load(model_path)
    ruta = ROOT / entrada["archivo"]
    if huella(ruta) != entrada["sha256"]:
        raise ValueError(f"{entrada['archivo']} no coincide con el sha256 del registry")

    pipe = joblib.load(ruta)          # PipelineFrio entrenado
    model = pipe.modelo_              # scaler -> selección -> modelo
    preprocesado = pipe               # derivadas + Preprocess ajustado
    print(f"Modelo cargado: {modelo} {entrada['version']} ({entrada['archivo']})")
    return model, preprocesado

def load_and_preprocess(filepath, preprocesado):
    """Carga el archivo Excel y aplica todo el preprocesamiento.
    Reusa el pipeline de preprocessing_pipeline.py: no lo reimplementa.
    Devuelve la matriz de caracteristicas y las fechas correspondientes.
    """
    # ... aplicar pipeline de preprocesamiento ...
    df = construir_dataset(rutas=filepath)
    X_processed = preprocesado._features(preprocesado._derivadas(df))
    fechas = X_processed.index        # una por fila de X, ya ordenadas
    return X_processed, fechas

def predict_consumption(filepath, modelo="mlp", version="latest"):
    """Funcion principal de prediccion."""
    model, preprocesado = load_model(modelo, version) # carga desde el registry
    X, fechas = load_and_preprocess(filepath, preprocesado)
    predictions = model.predict(X)
    # La prediccion es DIARIA: una fila por dia, no por hora.
    # 'fecha' es el dia D; la prediccion corresponde al consumo del dia D+1.
    return pd.DataFrame({
        'fecha': fechas,
        'prediccion_frio_kw': predictions
    })

if __name__ == "__main__":
    import train_model    # se importa a sí mismo: las clases quedan como train_model.PipelineFrio
    train_model.main()
    results = predict_consumption(sys.argv[1])
    results.to_csv('results/predicciones.csv', index=False)
    print("Predicciones generadas en results/predicciones.csv")