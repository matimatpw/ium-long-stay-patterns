"""Utilities for creating dataset CSVs used by the project.

This file provides:
- prepare_booking_data: returns booking-level DataFrame with stay duration flags.
- aggregate_listing_stats: aggregates bookings into listing-level statistics.
- save_listing_stats: saves the aggregated stats to CSV.
"""

from __future__ import annotations

from ium_long_stay_patterns.config import ProcessedCSV

import logging
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def prepare_booking_data(sessions_csv: str = ProcessedCSV.SESSIONS.path) -> pd.DataFrame:
    """Return a DataFrame of bookings with binary stay-length columns."""
    sessions = pd.read_csv(sessions_csv)

    bookings = sessions[sessions["action"] == "book_listing"].copy()

    bookings["booking_date"] = pd.to_datetime(bookings["booking_date"], errors="coerce")
    bookings["booking_duration"] = pd.to_datetime(bookings["booking_duration"], errors="coerce")

    # length in days (can be NaN if either date is missing)
    bookings["length_of_stay"] = (bookings["booking_duration"] - bookings["booking_date"]).dt.days

    bookings["is_long_stay"] = (bookings["length_of_stay"] >= 7).astype(int)
    # Define short stay as less than 7 days
    bookings["is_short_stay"] = (bookings["length_of_stay"] < 7).astype(int)

    return bookings


def aggregate_listing_stats(
    sessions_csv: str = ProcessedCSV.SESSIONS.path,
    listings_csv: str = ProcessedCSV.LISTINGS.path,
    bookings: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate booking-level data into listing-level statistics."""
    if bookings is None:
        bookings = prepare_booking_data(sessions_csv)

    # Use only bookings with a valid length_of_stay for statistics
    valid = bookings.dropna(subset=["length_of_stay"]).copy()

    agg = (
        valid.groupby("listing_id")
        .agg(
            total_bookings=("listing_id", "size"),
            min_stay=("length_of_stay", "min"),
            max_stay=("length_of_stay", "max"),
            avg_stay=("length_of_stay", "mean"),
            long_stay=("is_long_stay", "sum"),
            short_stay=("is_short_stay", "sum"),
        )
        .reset_index()
    )

    # Read listings to include listings with zero bookings
    listings = pd.read_csv(listings_csv, usecols=["id"]).rename(columns={"id": "listing_id"})
    listing_stats = listings.merge(agg, on="listing_id", how="left")

    # Replace NaNs for listings with no bookings and ensure types
    fill_values = {
        "total_bookings": 0,
        "min_stay": 0,
        "max_stay": 0,
        "avg_stay": 0.0,
        "long_stay": 0,
        "short_stay": 0
    }
    listing_stats = listing_stats.fillna(fill_values)

    # Cast numeric columns
    int_cols = ["total_bookings", "min_stay", "max_stay", "long_stay", "short_stay"]
    listing_stats[int_cols] = listing_stats[int_cols].astype(int)
    listing_stats["avg_stay"] = listing_stats["avg_stay"].round(2)

    # Reorder columns
    listing_stats = listing_stats[
        ["listing_id", "total_bookings", "min_stay", "max_stay", "avg_stay", "long_stay", "short_stay"]
    ]

    return listing_stats


def save_listing_stats(
    output_csv: str = ProcessedCSV.LISTINGS_STATS.path,
    sessions_csv: str = ProcessedCSV.SESSIONS.path,
    listings_csv: str = ProcessedCSV.LISTINGS.path,
    listing_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute (if needed) and save listing statistics to CSV."""
    if listing_stats is None:
        listing_stats = aggregate_listing_stats(sessions_csv=sessions_csv, listings_csv=listings_csv)

    listing_stats.to_csv(output_csv, index=False)
    logger.info("Saved listing stats to %s", output_csv)

    return listing_stats


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
#     save_listing_stats()