import os
import yaml
import pytest

def test_config_loads():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    assert "data" in config
    assert "model" in config
    assert "api" in config
    
    assert config["data"]["raw_path"] == "data/raw/amazon_reviews.parquet"
    assert config["model"]["experiment_name"] == "amazon_recommendation"

def test_pipeline_modules_importable():
    """Ensure that all pipeline modules can be imported without syntax errors."""
    import src.ingestion.data_loader
    import src.eda.eda
    import src.preprocessing.cleaner
    import src.features.feature_engineering
    import src.models.train
    
    # Check if run functions are defined
    assert callable(src.ingestion.data_loader.run_ingestion)
    assert callable(src.eda.eda.run_eda)
    assert callable(src.preprocessing.cleaner.run_cleaning)
    assert callable(src.features.feature_engineering.run_feature_engineering)
    assert callable(src.models.train.main)

def test_directory_structure():
    """Ensure that the necessary data and model directories exist."""
    root = os.path.dirname(os.path.dirname(__file__))
    
    dirs_to_check = [
        "data/raw",
        "data/processed",
        "data/features",
        "models",
        "mlruns",
        "src/utils"
    ]
    
    for d in dirs_to_check:
        path = os.path.join(root, d)
        assert os.path.isdir(path), f"Directory {path} should exist"
