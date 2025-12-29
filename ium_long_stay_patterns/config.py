from pathlib import Path
from enum import Enum

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
