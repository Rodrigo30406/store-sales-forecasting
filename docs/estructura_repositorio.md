# Estructura del Repositorio: store-sales-forecasting

Este documento describe la estructura propuesta para un proyecto de forecasting de ventas orientado a producción. La organización está pensada para escalar, facilitar pruebas, integración con MLOps, y mantener claridad entre las distintas fases del flujo de trabajo.

## Raíz del proyecto

```
store-sales-forecasting/
├── data/
├── notebooks/
├── dashboards/
├── reports/
├── submissions/
├── src/
├── tests/
├── scripts/
├── requirements.txt
├── pyproject.toml
├── README.md
```

---

## Carpetas y Archivos

### 💾 `data/`
- **raw/**: Datos originales sin procesar.
- **interim/**: Datos transformados temporalmente (por ejemplo, normalizados pero sin features).
- **processed/**: Datos listos para entrenar modelos.

### 📃 `notebooks/`
- Exploración, validación rápida, pruebas con modelos.

### 📊 `dashboards/`
- Dashboards generados con Streamlit, Plotly u otras herramientas para presentación.

### 📄 `reports/`
- Reportes PDF u otros archivos de resultados.

### 🔹 `submissions/`
- Archivos `.csv` listos para subir a Kaggle u otras plataformas.

### 🔬 `src/`
**Carpeta principal de código fuente**:

- `config/`
  - `config.yaml`: Archivo central de configuración.
  - `config_loader.py`: Clase `ConfigLoader` y objeto global `CONFIG` para cargar el YAML.

- `data/`
  - `load.py`: Carga de datasets (train/test/etc.) según `config.yaml`.
  - `validate.py`: Validación de datos crudos (estructuras, tipos, vacíos).

- `features/`
  - `custom_transformers.py`: Transformers personalizados (`DatePartAdder`, `CyclicEncoder`, etc.)

- `pipelines/`
  - `pipeline_builder.py`: Ensambla el pipeline de transformación.

- `models/`
  - `trainer.py`: Entrenamiento de modelos (LSTM, LightGBM, etc.).
  - `predictor.py`: Carga modelo y predice sobre nuevos datos.
  - `model_utils.py`: Funciones para guardar, cargar y gestionar modelos.

- `evaluation/`
  - `metrics.py`: Cálculo de métricas (MAE, RMSE, etc.).

- `utils/`
  - `io.py`: Funciones auxiliares de entrada/salida.

### 🧰 `tests/`
- `test_pipeline.py`: Verifica que el pipeline funcione correctamente.
- `test_transformers.py`: Pruebas unitarias de cada transformer.
- `test_config.py`: Comprueba que `config.yaml` se cargue y contenga lo necesario.

### 🔧 `scripts/`
- `run_training.py`: Ejecuta el flujo completo de entrenamiento.
- `predict_batch.py`: Predice sobre datos nuevos, genera archivos de salida.

---

## Uso de `CONFIG`

```python
from src.config.config_loader import CONFIG
train_path = CONFIG["paths"]["train"]
```
Evita repetir lógica de carga y mantiene consistencia en todo el proyecto.

---

## Ventajas

- Modular y clara.
- Compatible con pruebas unitarias y MLOps.
- Escalable para nuevos modelos o etapas.
- Configurable sin cambiar el código.

---

Este documento puede versionarse como parte del repositorio en `/docs/estructura_repositorio.md` para consulta y onboarding.