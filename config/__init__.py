import yaml
from pathlib import Path

path = Path(__file__).parent

config = yaml.safe_load(open((path / 'config.yaml')))

__all__ = [config]
