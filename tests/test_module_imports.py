import importlib


def test_analysis_package_imports():
    importlib.import_module("src.analysis")


def test_attack_training_module_imports():
    importlib.import_module("src.attacks.train")
