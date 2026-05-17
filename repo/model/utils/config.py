import yaml
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")
    return config


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    if model_name not in config:
        raise ValueError(
            f"Model '{model_name}' not found in config. Choose from: gan, vae, transformer, diffusion"
        )

    model_cfg = config[model_name].copy()
    model_cfg["batch_size"] = config["training"]["batch_size"]
    model_cfg["epochs"] = config["training"]["epochs"]
    model_cfg["seed"] = config["seed"]
    return model_cfg