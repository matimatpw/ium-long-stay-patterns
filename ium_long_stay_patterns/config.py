from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"


class ProcessedCSV(Enum):
    """Paths to processed CSV files inside `data/processed`."""

    SESSIONS = PROCESSED_DATA_DIR / "sessions.csv"
    REVIEWS = PROCESSED_DATA_DIR / "reviews.csv"
    LISTINGS = PROCESSED_DATA_DIR / "listings.csv"
    USERS = PROCESSED_DATA_DIR / "users.csv"
    LISTINGS_STATS = PROCESSED_DATA_DIR / "listings_stats.csv"

    @property
    def path(self) -> Path:
        """Return the `Path` for this processed CSV file."""
        return self.value

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.path)


MODELS_DIR = PROJ_ROOT / "models"
SAVED_MODELS_DIR = PROJ_ROOT / "saved_models"

TEST_DATA_DIR = PROJ_ROOT / "prediction_service" / "test_data"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


def set_seed(seed: int, deterministic: bool = True, verbose=False) -> None:
    """Set global seeds for Python, NumPy and PyTorch.

    This ensures reproducibility across the project where deterministic
    behavior is desired. If PyTorch is not available, NumPy and Python
    random will still be seeded.

    Args:
        seed: integer seed
        deterministic: if True, set cuDNN to deterministic mode (may reduce perf)
    """
    import os
    import random as _random

    import numpy as _np

    try:
        import torch as _torch
    except Exception:
        _torch = None

    _random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        if deterministic:
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    if verbose:
        logger.info(f"Global seed set to {seed} (deterministic={deterministic})")
