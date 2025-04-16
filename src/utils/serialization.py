import pickle
from pathlib import Path
from typing import Any

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