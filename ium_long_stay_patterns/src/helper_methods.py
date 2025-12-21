import pandas as pd
import matplotlib.pyplot as plt

def prepare_booking_data(sessions_csv="data/raw/sessions.csv"):
    """
    Return a DataFrame of bookings with a binary 'is_long_stay' column:
    1 if length_of_stay >= 7 days, else 0.
    This is booking-level (each row is a booking).
    """
    sessions = pd.read_csv(sessions_csv)
    bookings = sessions[sessions["action"] == "book_listing"].copy()
    bookings["booking_date"] = pd.to_datetime(bookings["booking_date"], errors="coerce")
    bookings["booking_duration"] = pd.to_datetime(bookings["booking_duration"], errors="coerce")
    bookings["length_of_stay"] = (bookings["booking_duration"] - bookings["booking_date"]).dt.days
    bookings["is_long_stay"] = (bookings["length_of_stay"] >= 7).astype(int)
    # print(bookings.head())

    return bookings

def prepare_listing_data(listings_csv="data/raw/listings.csv", sessions_csv="data/raw/sessions.csv"):
    """
    Return a DataFrame aggregated per listing with a binary 'is_long_stay' label.
    Label options can be changed — here we label a listing as long-stay if its
    average stay >= 7 days.
    """
    listings = pd.read_csv(listings_csv)
    bookings = prepare_booking_data(sessions_csv)
    agg = bookings.groupby("listing_id").agg(
        total_bookings=("booking_id", "count"),
        min_stay=("length_of_stay", "min"),
        max_stay=("length_of_stay", "max"),
        avg_stay=("length_of_stay", "mean"),
        stays_gte_7=("length_of_stay", lambda x: (x >= 7).sum())
    ).reset_index()
    data = listings.merge(agg, left_on="id", right_on="listing_id", how="left")
    # Fill NaNs for listings with no bookings (optional)
    data[["total_bookings", "min_stay", "max_stay", "avg_stay", "stays_gte_7"]] = \
        data[["total_bookings", "min_stay", "max_stay", "avg_stay", "stays_gte_7"]].fillna(0)
    # Define target: listing is long-stay if average stay >= 7
    data["is_long_stay"] = (data["avg_stay"] >= 7).astype(int)
    # print(data.head())
    return data

def plot_long_stay_distribution(df, kind="booking", save_path=None, ax=None):
    """
    Plot distribution of 'is_long_stay' (0 vs 1) using matplotlib.
    - df: DataFrame that must contain 'is_long_stay' column.
    - kind: just for title, e.g., 'booking' or 'listing'
    - save_path: if provided, figure will be saved to this path.
    - ax: optional matplotlib Axes to draw into.
    """
    if "is_long_stay" not in df.columns:
        raise ValueError("DataFrame must contain 'is_long_stay' column")

    counts = df["is_long_stay"].value_counts().sort_index()
    labels = ["< 7 days (0)", ">= 7 days (1)"]
    # ensure both labels exist
    counts = counts.reindex([0, 1], fill_value=0)

    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        own_fig = True

    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"], alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of long_stay (kind={kind})")
    ax.set_ylim(0, counts.values.max() * 1.15 if counts.values.max() > 0 else 1)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        pct = (value / total * 100) if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.01, f"{value}\n{pct:.1f}%",
                ha="center", va="bottom", fontsize=9)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)

    # if own_fig:
    #     plt.show()
