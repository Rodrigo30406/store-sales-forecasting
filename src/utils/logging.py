import logging
from pathlib import Path
import mlflow

def setup_logger(name: str = 'store_sales', log_file: str = "logs/default.log", level=logging.INFO) -> logging.Logger:
    """
    Configura y devuelve un logger con consola y archivo.

    Args:
        name (str): Nombre del logger.
        log_file (str): Ruta al archivo de log.
        level (int): Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita duplicar mensajes si el logger ya tiene handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Asegura que la carpeta exista
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Formato para consola
    formatter_console = logging.Formatter(
        "%(filename)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Formato para archivo
    formatter_file = logging.Formatter(
        "%(asctime)s | %(filename)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter_console)
    logger.addHandler(console_handler)

    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter_file)
    logger.addHandler(file_handler)

    return logger


def log_pipeline_readable(pipeline):
    """
    Registra en MLflow una representación legible del pipeline en formato JSON.
    """
    pipeline_dict = {name: step.__class__.__name__ for name, step in pipeline.steps}
    mlflow.log_dict(pipeline_dict, "pipeline_structure.json")
