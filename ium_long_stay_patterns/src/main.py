from helper_methods import (
    plot_long_stay_distribution,
    prepare_booking_data,
    prepare_listing_data,
)
import pandas as pd

# chunksize = 100000  # 100k rows at a time
# for chunk in pd.read_csv("data/raw/sessions.csv", chunksize=chunksize):
#     print(chunk.head())
#     # do analysis on this chunk


# Example usage:
# Booking-level distribution
bookings = prepare_booking_data()
plot_long_stay_distribution(
    bookings, kind="booking", save_path="reports/figures/booking_long_stay_dist.png"
)

# # Listing-level distribution
listings = prepare_listing_data()
plot_long_stay_distribution(
    listings, kind="listing", save_path="reports/figures/listing_long_stay_dist.png"
)
