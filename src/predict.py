"""Predicción del frío (kW) del día siguiente a partir de un Excel de planta.

Uso:
    python src/predict.py archivo.xlsx                           # mlp, última versión del registro
    python src/predict.py archivo.xlsx --modelo mlp --version v0.1.0
    python src/predict.py archivo.xlsx --modelo lasso --salida results/pred_lasso.csv

El Excel es como los de data/raw ("Totalizadores Planta"): registros horarios en varias hojas. Hojas y
columnas se buscan por nombre; las que sobran se ignoran y, si falta algo necesario, se dice qué falta.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import *
from train_model import *
from train_model import _sin_avisos   # `import *` no trae los nombres que empiezan con _
RUTA_PREDICCIONES = ROOT / "results" / "predicciones.csv"


def _clave_version(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.removeprefix("v").split("."))


def load_model(modelo="mlp", version="latest"):
    """Carga el modelo y los preprocesadores desde el registro.
    'latest' carga el modelo con la version mas alta de ese tipo de modelo (mlp, lasso).
    Devuelve (model, preprocesado): `preprocesado.transformar(df)` arma las variables que espera `model`.
    """
    registry = cargar_registro(RUTA_REGISTRO_MODELOS)
    if not registry:
        raise FileNotFoundError(f"No existe o está vacío el registro de modelos {RUTA_REGISTRO_MODELOS}")
    entradas = [e for e in registry if e["modelo"] == modelo]
    if not entradas:
        raise ValueError(f"No hay ningún modelo {modelo!r} en el registro "
                         f"(hay: {sorted({e['modelo'] for e in registry})})")
    if version == "latest":
        entrada = max(entradas, key=lambda e: _clave_version(e["version"]))
    else:
        version = normalizar_version(version)
        entrada = next((e for e in entradas if e["version"] == version), None)
        if entrada is None:
            raise ValueError(f"No existe {modelo} {version} en el registro "
                             f"(versiones de {modelo}: {[e['version'] for e in entradas]})")

    model_path = ROOT / entrada["archivo"]
    if not model_path.exists():
        raise FileNotFoundError(f"Falta el archivo {entrada['archivo']} de {modelo} {entrada['version']}")
    if huella(model_path) != entrada["sha256"]:
        raise ValueError(f"{entrada['archivo']} no coincide con el sha256 del registro: no es el modelo registrado")

    objeto = joblib.load(model_path)
    if entrada.get("tipo") == TIPO_PIPELINE:
        # PipelineFrio: trae adentro el preprocesamiento ajustado y el modelo
        model, preprocesado = objeto.modelo_, objeto
    else:
        # Pipeline entrenado sobre dataset_final.csv: el preprocesamiento se rearma desde el linaje de ese dataset
        model, preprocesado = objeto, preprocesado_registrado(modelo, entrada["version"])
    print(f"Modelo: {modelo} {entrada['version']} ({entrada['archivo']})")
    return model, preprocesado


def load_and_preprocess(filepath, preprocesado, necesarias):
    """Carga el archivo Excel y aplica todo el preprocesamiento.
    Reusa las funciones de preprocess.py y el preprocesamiento del modelo: no lo reimplementa.
    Devuelve la matriz de caracteristicas y las fechas correspondientes.
    `necesarias`: columnas de planta sin las que el modelo no puede predecir.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"No existe el archivo {filepath}")
    try:
        pd.ExcelFile(filepath).close()
    except (ValueError, zipfile.BadZipFile, OSError) as e:
        raise ValueError(f"{filepath}: no se pudo abrir como Excel (.xlsx): {e}") from e
    # solo las hojas de datos, buscadas por nombre: las demás hojas del archivo se ignoran
    df = juntar_datasets([procesar_excel(filepath, hojas=HOJAS_DATOS)])
    if df.empty:
        raise ValueError(f"{filepath}: no tiene ningún día completo (25 registros horarios y cierre >= 23:30)")

    if faltan := [c for c in necesarias if c not in df.columns]:
        with pd.ExcelFile(filepath) as libro:
            en_archivo = {str(n).strip().lower() for n in libro.sheet_names}
        hojas_ausentes = [h for h in HOJAS_DATOS if h.lower() not in en_archivo]
        raise ValueError(
            f"{filepath}: faltan columnas que el modelo necesita: {faltan}."
            + (f" Hojas esperadas que no están en el archivo: {hojas_ausentes}." if hojas_ausentes
               else " Las hojas esperadas están, pero ninguna tiene esas columnas."))

    # por nombre: solo las columnas que conoce el preprocesamiento; lo que sobra en el Excel se descarta
    conocidas = [c for c in preprocesado.columnas_entrada() if c not in COLS_SIN_LIMITES]
    presentes = [c for c in conocidas if c in df.columns]
    ausentes = [c for c in conocidas if c not in df.columns]
    df = completar_dias(convertir_object_a_numerico(df[[COL_FECHA, *presentes]]))
    if ausentes:
        print(f"Aviso: {len(ausentes)} de {len(conocidas)} columnas que conoce el preprocesamiento no están en el "
              f"Excel y el modelo no las usa: se completan con el imputador (p. ej. {ausentes[:3]})")
        df = df.reindex(columns=[*df.columns, *ausentes])   # vacías: las completa el imputador

    try:
        df = agregar_clima(df)
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo bajar el clima de Open-Meteo para {df[COL_FECHA].min():%Y-%m-%d} a "
                           f"{df[COL_FECHA].max():%Y-%m-%d}: {e}") from e
    clima = [c for c in COLS_SIN_LIMITES if c in df.columns]
    if n := int(df[clima].isna().any(axis=1).sum()):
        print(f"Aviso: {n} días sin datos de clima (Open-Meteo todavía no los publica): se imputan")
    df = agregar_mes_ciclico(df)
    if len(df) < DIAS_HISTORIA:
        print(f"Aviso: el archivo tiene {len(df)} días; con menos de {DIAS_HISTORIA} los rezagos y promedios "
              f"móviles quedan incompletos (se imputan)")

    X_processed = preprocesado.transformar(df)
    fechas = X_processed.index
    return X_processed, fechas


def predict_consumption(filepath, modelo="mlp", version="latest"):
    """Funcion principal de prediccion."""
    model, preprocesado = load_model(modelo, version)   # carga desde el registry
    necesarias = preprocesado.columnas_necesarias(columnas_elegidas(model))
    X, fechas = load_and_preprocess(filepath, preprocesado, necesarias)
    with _sin_avisos():
        predictions = model.predict(X[list(model.feature_names_in_)])
    # La prediccion es DIARIA: una fila por dia, no por hora.
    # 'fecha' es el dia D; la prediccion corresponde al consumo del dia D+1.
    return pd.DataFrame({
        'fecha': fechas,
        'prediccion_frio_kw': predictions
    })


def main():
    parser = argparse.ArgumentParser(description="Predice el frío (kW) del día siguiente para un Excel de planta.")
    parser.add_argument("filepath", help="Excel de planta (.xlsx)")
    parser.add_argument("--modelo", default="mlp", help="tipo de modelo del registro (por defecto %(default)s)")
    parser.add_argument("--version", default="latest", help="'latest' o una versión, p. ej. v1.0.0")
    parser.add_argument("--salida", default=str(RUTA_PREDICCIONES), help="csv de salida (por defecto %(default)s)")
    args = parser.parse_args()
    try:
        results = predict_consumption(args.filepath, args.modelo, args.version)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"ERROR: {e}")
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(salida, index=False)
    print(f"Predicciones generadas en {salida} ({len(results)} días, "
          f"{results['fecha'].min():%Y-%m-%d} a {results['fecha'].max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
