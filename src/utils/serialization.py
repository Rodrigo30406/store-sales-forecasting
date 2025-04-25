from src.config import CONFIG
import pandas as pd
import pickle
from pathlib import Path
import joblib
from typing import Any
import os
import logging

logger = logging.getLogger(CONFIG.logger_name)

def save_object(obj: Any, path: str) -> None:
    """
    Guarda cualquier objeto Python serializable en un archivo .pkl usando pickle.

    Args:
        obj (Any): Objeto a guardar.
        path (str): Ruta donde se guardará el archivo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # Crea directorios si no existen
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str) -> Any:
    """
    Carga un objeto serializado desde un archivo .pkl.

    Args:
        path (str): Ruta al archivo pickle.

    Returns:
        Any: Objeto cargado desde el archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        Exception: Si ocurre un error al deserializar.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {e}")


def save_model(model, path: str) -> None:
    """
    Guarda un modelo entrenado usando joblib.

    Parameters:
        model: El objeto del modelo entrenado (ej. sklearn, LGBM, etc.)
        path (str): Ruta completa donde se desea guardar el archivo (incluye nombre y extensión .pkl)

    Raises:
        ValueError: Si no se especifica un path válido.
        Exception: Si ocurre un error al guardar el modelo.
    """
    
    # Crear carpeta si no existe
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    try:
        logger.info("💾 Guardando modelo entrenado...")
        joblib.dump(model, path)
        logger.info(f"✅ Modelo guardado exitosamente en: {path}")
    except Exception as e:
        raise Exception(f"❌ Error al guardar el modelo: {e}")


def load_model(path: str):
    """
    Carga un modelo previamente guardado con joblib.

    Parameters:
        path (str): Ruta completa del archivo .pkl a cargar

    Returns:
        El modelo cargado desde el archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        Exception: Si ocurre un error al cargar el modelo.
    """
    
    try:
        model = joblib.load(path)
        logger.info(f"✅ Modelo cargado exitosamente desde: {path}")
        return model
    except Exception as e:
        raise Exception(f"❌ Error al cargar el modelo: {e}")


def save_predictions(ids, predictions, output_path: str):
    """
    Guarda las predicciones en un archivo CSV con formato estándar.

    Args:
        ids (array-like): Índices o identificadores únicos (ej. df_test.index).
        predictions (array-like): Valores predichos de la variable target.
        output_path (str): Ruta donde se guardará el archivo CSV.
    """
    # Asegurar que el directorio existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Crear DataFrame y guardar
    df_output = pd.DataFrame({
        "id": ids,
        "sales": predictions
    })
    df_output.to_csv(output_path, index=False)

    logger.info(f"✅ Predicciones guardadas en {output_path}")
