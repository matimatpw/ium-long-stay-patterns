from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from ium_long_stay_patterns.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ProcessedCSV
from ium_long_stay_patterns.src.helpers.create_listing_stats_dataset import (
    save_listing_stats,
)

app = typer.Typer()


def create_listing_stats_dataset(
    sessions_csv: str = ProcessedCSV.SESSIONS.path,
    listings_csv: str = ProcessedCSV.LISTINGS.path,
    output_csv: str = ProcessedCSV.LISTINGS_STATS.path,
) -> None:
    """Create and save the listing stats dataset."""
    logger.info("Creating listing stats dataset...")
    save_listing_stats()
    logger.info(f"Listing stats dataset saved to {output_csv}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    listing_stats: bool = True,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    create_listing_stats_dataset()
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
