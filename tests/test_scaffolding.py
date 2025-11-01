"""Lightweight smoke tests for the project layout."""
import importlib

import pytest


@pytest.mark.parametrize(
    "module_name", ["src.processing", "src.methods", "src.visualizations"]
)
def test_modules_import(module_name):
    module = importlib.import_module(module_name)
    assert module.__all__  # ensure public exports are defined


def test_config_paths_are_paths():
    from src import config

    assert config.DATA_DIR.name == "data"
