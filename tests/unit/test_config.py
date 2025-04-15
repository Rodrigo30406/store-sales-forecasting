import pytest
from src.config import CONFIG

def test_config_loads_as_dict():
    config_data = CONFIG.raw()
    assert isinstance(config_data, dict)
    assert "paths" in config_data
    assert "variables" in config_data

def test_train_path_exists():
    path = CONFIG.get_path("train")
    assert isinstance(path, str)
    assert path.startswith("http")  # porque lo cargas de huggingface

def test_flattened_categoricals():
    cats = CONFIG.get_variable("categorical", flatten=True)
    assert isinstance(cats, list)
    assert "family" in cats
    assert "type" in cats

def test_static():
    static = CONFIG.get_variable("categorical", subkey="static", flatten=False)
    assert isinstance(static, dict)
    assert list(static.keys()) == ['default', 'added']

def test_static_flattened():
    static = CONFIG.get_variable("categorical", subkey="static", flatten=True)
    assert static == ["family", "store_nbr", "city", "state", "type", "cluster"]

def test_dynamic_flattened():
    static = CONFIG.get_variable("categorical", subkey="dynamic", flatten=True)
    assert static == []

def test_invalid_path_returns_none():
    assert CONFIG.get_path("not_existing") is None