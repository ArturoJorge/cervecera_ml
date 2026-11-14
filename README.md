# cervecera_ml
Proyecto final para Laboratorio de Datos II.
## Grupo
- **Detzel Kolman** Axel Nicolas
- **Carrara** Ignacio
- **Jorge** Arturo

## Objetivo
Predecir el consumo energético del sistema de refrigeración, **Frío (kW) del día siguiente**, en una planta cervecera mexicana. Este README explica cómo preparar el entorno y cómo generar predicciones con `src/predict.py`.

## 1. Preparar el entorno
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/ArturoJorge/cervecera_ml.git
   cd cervecera_ml
   ```
2. **Crear el entorno e instalar las dependencias** (Python 3.12)
   ```bash
   conda create -n cervecera_env python=3.12
   conda activate cervecera_env
   pip install -r requirements.txt
   ```
   Con `venv`: `python -m venv .venv`, activarlo y ejecutar `pip install -r requirements.txt`.

Los modelos entrenados (`models/*.joblib`) y su registro (`models/model_registry.json`) ya están en el repositorio: no hace falta entrenar nada para predecir.

## 2. Predecir con `predict.py`
Desde la raíz del proyecto, con el entorno activado:
```bash
python src/predict.py "ruta/al/archivo.xlsx"
```
Esto usa la última versión del modelo `mlp` y escribe `results/predicciones.csv`.

### Opciones
| Opción | Por defecto | Qué hace |
|---|---|---|
| `--modelo` | `mlp` | Tipo de modelo del registro (`mlp` o `lasso`). |
| `--version` | `latest` | `latest` usa la versión más alta de ese tipo de modelo; también se puede pedir una específica, por ejemplo `v1.0.0`. |
| `--salida` | `results/predicciones.csv` | Ruta del `.csv` de salida. |

```bash
python src/predict.py "ruta/al/archivo.xlsx" --modelo mlp --version v1.0.0
python src/predict.py "ruta/al/archivo.xlsx" --modelo lasso --salida results/pred_lasso.csv
```

### Archivo de entrada
Un Excel `.xlsx` con el formato de los "Totalizadores Planta": registros horarios, con las columnas `dia` y `hora` en cada hoja de datos.

- Las hojas y las columnas se buscan **por nombre** (sin distinguir mayúsculas ni espacios al principio o al final). El orden no importa y las hojas o columnas de más se ignoran.
- Las columnas que el modelo `mlp` necesita sí o sí son (otros modelos pueden pedir más):

  | Hoja | Columnas |
  |---|---|
  | `Consolidado EE` | `frio (kw)`, `planta (kw)`, `sala maq (kw)`, `servicios (kw)` |
  | `Consolidado Produccion` | `hl cerveza envasada` |

  Si falta alguna, el script termina con un mensaje que dice qué columnas y qué hojas faltan.
- Solo se usan los días completos: 25 registros (24 horarios más el cierre) y un registro de cierre a las 23:30 o después.
- Conviene que el archivo tenga al menos 15 días seguidos: el modelo usa valores de hasta 14 días atrás, y las predicciones de los primeros días salen con esa historia incompleta.

### Archivo de salida
Un `.csv` con una fila por día:

| Columna | Contenido |
|---|---|
| `fecha` | Día D de los datos. |
| `prediccion_frio_kw` | Frío (kW) predicho para el día D+1. |

### A tener en cuenta
- **Conexión a internet:** el clima de la planta se baja de Open-Meteo para las fechas del Excel. Las fechas entre 2020-07-01 y 2023-10-25 ya están guardadas en `data/clima/clima_tecate.csv` y no necesitan conexión.
- **Versiones `v0.x`:** su preprocesamiento se reconstruye a partir de `data/clean/df_total.xlsx`, que no está en el repositorio. La versión por defecto (`mlp` `latest`) no lo necesita.
