import os
import yaml
from pathlib import Path

def load_config():
    """Load the central configuration file."""
    # Find the absolute path to the project root (assumes utils/ is inside src/ and src/ is in root)
    root_dir = Path(__file__).resolve().parent.parent.parent
    config_path = root_dir / "configs" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config, root_dir
