import yaml
from pathlib import Path

class ConfigLoader:
    """
    Carga centralizada del archivo config.yaml.
    Acceso mediante métodos y `.raw()` (sin __getitem__).
    """
    def __init__(self, config_path: str = "configs/project_config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        self._config = self._load_config()
        self.logger_name = self._config["logger"]

    def _load_config(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def raw(self):
        """Devuelve todo el YAML como diccionario."""
        return self._config

    def get_path(self, key: str):
        return self._config["paths"].get(key)

    def get_variable(self, key: str, subkey: str = None, flatten: bool = False):
        """
        Accede a una variable del bloque 'variables' del config.
        Si flatten=True y subkey es None, recorre recursivamente y une listas.
        """
        section = self._config.get("variables", {}).get(key, {})

        if subkey:
            section = section.get(subkey, {})

        if flatten:
            # Flatten profundo solo si no se especifica subkey
            def collect_lists(obj):
                if isinstance(obj, list):
                    return obj
                elif isinstance(obj, dict):
                    result = []
                    for v in obj.values():
                        result.extend(collect_lists(v))
                    return result
                else:
                    return []

            return collect_lists(section)

        return section
    
    def get_active_model_name(self):
        return self.raw().get("model", {}).get("active", "")

    def get_model_params(self):
        model_name = self.get_active_model_name()
        return self.raw().get("model", {}).get(model_name, {}).get("params", {})

    def get_search_space(self):
        model_name = self.get_active_model_name()
        return self.raw().get("model", {}).get(model_name, {}).get("search_space", {})


# Instancia global para importar en todo el proyecto
CONFIG = ConfigLoader()