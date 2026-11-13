# cervecera_ml
Proyecto final para Laboratorio de Datos II.
## Grupo
- **Detzel Kolman** Axel Nicolas
- **Carrara** Ignacio
- **Jorge** Arturo

## Objetivo
Desarrollar un pipeline de machine learning completo y reproducible para predecir el consumo energético del sistema de refrigeración **Frío (kW) del día siguiente** en una planta cervecera mexicana. Además de la precisión del modelo, el proyecto evalúa la aplicación de buenas prácticas de MLOps para garantizar la trazabilidad, la colaboración y la gestión del ciclo de vida del proyecto.

## Estructura
```
data/
  raw/     # Excel originales (no versionados, ver paso 3)
  clean/   # dataset limpio generado por el el codigo (no versionado)
  logs/    # data_log.json: registro de versiones del dataset limpio
notebooks/
  Eda.ipynb                # carga, limpieza, versionado y análisis exploratorio
  preprocesamiento.ipynb   # features, splits y dataset final
```

## Cómo ejecutarlo
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/ArturoJorge/cervecera_ml.git
   cd cervecera_ml
   ```
2. **Crear el entorno e instalar dependencias** (Python 3.11 o superior)
   ```bash
   conda create -n cervecera_env python=3.12
   conda activate cervecera_env
   pip install -r requirements.txt
   ```
   Con `venv`: `python -m venv .venv`, actívalo y luego ejecuta `pip install -r requirements.txt`.
3. **Colocar los datos crudos** en `data/raw/` con estos nombres exactos (no están en el repo por su tamaño):
   - `Totalizadores Planta - 2021_2023.xlsx`
   - `Totalizadores Planta 2020_2022.xlsx`
   - `Totalizadores Planta 2022_2023.xlsx`
4. **Abrir el notebook**
   ```bash
   jupyter lab notebooks/Eda.ipynb
   ```
5. **Ejecutar todas las celdas en orden** (*Run → Run All Cells*). El notebook:
   - carga y limpia los tres Excel y los une en `df_total`;
   - guarda el resultado en `data/clean/df_total.xlsx`;
   - registra la versión (hash SHA-256 y transformaciones) en `data/logs/data_log.json`;
   - hace el análisis exploratorio: nulos, valores negativos, correlaciones con `frio_mañana (kw)` y outliers.
