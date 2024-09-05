from pathlib import Path
import yaml
import config as cfg

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config(file_path: Path, file_name: str) -> dict:
    with open(Path.joinpath(file_path, file_name), 'r') as stream:
        settings = yaml.safe_load(stream)
    return settings

def load_model_config(dataset_name: str, model_name: str) -> dict:
    if model_name == "hierarch":
        config = load_config(cfg.HIERARCH_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    elif model_name == "mensa":
        config = load_config(cfg.MENSA_CONFIGS_DIR, f"{dataset_name.lower()}.yaml")
    else:
        raise ValueError("Invalid config name")
    return config