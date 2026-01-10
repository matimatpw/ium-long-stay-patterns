import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

listings = pd.read_csv("data/raw/listings.csv")
sessions = pd.read_csv("data/raw/sessions.csv")

# pusta kolumna
listings = listings.drop(columns=["calendar_updated"])


bookings = sessions[sessions["action"] == "book_listing"].copy()


bookings["booking_date"] = pd.to_datetime(bookings["booking_date"])
bookings["booking_duration"] = pd.to_datetime(bookings["booking_duration"])

# długość pobytu = data końcowa - data początkowa
bookings["length_of_stay"] = (bookings["booking_duration"] - bookings["booking_date"]).dt.days


agg = (
    bookings.groupby("listing_id")
    .agg(
        total_bookings=("booking_id", "count"),
        min_stay=("length_of_stay", "min"),
        max_stay=("length_of_stay", "max"),
        avg_stay=("length_of_stay", "mean"),
        stays_gte_7=("length_of_stay", lambda x: (x >= 7).sum()),
    )
    .reset_index()
)

data = listings.merge(agg, left_on="id", right_on="listing_id", how="left")
print(data.head())


booking_vars = ["total_bookings", "min_stay", "max_stay", "avg_stay", "stays_gte_7"]

# only numeric columns from listings
listing_numeric = listings.select_dtypes(include="number").columns.tolist()

corr_matrix = data[listing_numeric + booking_vars].corr()
corr_filtered = corr_matrix.loc[listing_numeric, booking_vars]

plt.figure(figsize=(22, 10))
sns.heatmap(corr_filtered, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Korelacja: Atrybuty listings (Y) vs Statystyki rezerwacji (X)")
plt.xlabel("Statystyki rezerwacji")
plt.ylabel("Atrybuty listings")
plt.savefig("reports/figures/correlation_matrix.png")
