"""Read shared configuration without selecting experimental decisions."""

from pathlib import Path

def load_config(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Install the project dependencies first: python -m pip install -r requirements.txt"
        ) from exc
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping.")
    return config
