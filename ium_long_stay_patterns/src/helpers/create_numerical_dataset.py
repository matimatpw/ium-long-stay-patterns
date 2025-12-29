import pandas as pd
import numpy as np
import ast
from ium_long_stay_patterns.config import ProcessedCSV


DEFAULT_NUMERICAL_COLUMNS = [
    'id', 'host_id', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
    'host_listings_count', 'host_total_listings_count', 'host_verifications',
    'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',
    'number_of_reviews',
    'instant_bookable', 'calculated_host_listings_count',
    'reviews_per_month'
]

def create_numerical_dataset(listings_csv, columns=DEFAULT_NUMERICAL_COLUMNS, dropna_columns_that_are_all_nan=True, strategy=False):
    """
    Read listings CSV and return a DataFrame containing only numeric columns.

    - Parses and converts: price -> float, host_is_superhost -> {1,0}, host_acceptance_rate
      and host_response_rate -> floats in [0,1], host_verifications -> count,
      instant_bookable -> {1,0}.
    - Any other convertible columns (review_scores_*) will be coerced to numeric.
    - By default drops columns that are all NaN after conversion.

    Returns
    -------
    pandas.DataFrame
        DataFrame with numeric dtypes only (safe to pass into sklearn train_test_split).
    """
    # Load the dataset
    df = pd.read_csv(listings_csv)

    # Pre-select columns if provided (ensuring they exist in the CSV)
    if columns:
        existing_cols = [c for c in columns if c in df.columns]
        df = df[existing_cols].copy()

    # 1. Convert Price: remove '$' and ',' then convert to float
    if 'price' in df.columns:
        df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)

    # 2. Convert Boolean flags ('t'/'f') to integers (1/0)
    bool_cols = ['host_is_superhost', 'instant_bookable', 'host_has_profile_pic',
                 'host_identity_verified', 'has_availability']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)

    # 3. Convert Rates: "86%" -> 0.86
    rate_cols = ['host_response_rate', 'host_acceptance_rate']
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].str.replace('%', '', regex=False).astype(float) / 100.0

    # 4. Convert host_verifications: count items in the string list "['email', 'phone']"
    if 'host_verifications' in df.columns:
        def count_verifications(x):
            try:
                # ast.literal_eval safely evaluates a string into a Python list
                return len(ast.literal_eval(x))
            except (ValueError, SyntaxError):
                return 0
        df['host_verifications'] = df['host_verifications'].apply(count_verifications)

    # 5. Coerce all columns to numeric, setting unparseable values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # 6. Filter to keep only numeric dtypes (float64 and int64)
    df = df.select_dtypes(include=[np.number])

    # 7. Drop columns that are completely empty after cleaning
    if dropna_columns_that_are_all_nan:
        df = df.dropna(axis=1, how='all')

    if strategy:
        return strategy_for_nans(df.copy())

    return df



def strategy_for_nans(df):
    df_numeric = df.copy()
    # 1. Cena - usuwamy wiersze, gdzie brakuje ceny (Target)
    df_numeric = df_numeric.dropna(subset=['price'])

    # 2. Recenzje - brak recenzji to po prostu 0 recenzji na miesiąc
    df_numeric['reviews_per_month'] = df_numeric['reviews_per_month'].fillna(0)

    # 3. Fizyczne cechy - wypełniamy medianą
    for col in ['beds', 'bedrooms', 'bathrooms']:
        if col in df_numeric.columns:
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    # 4. Wskaźniki hosta - wypełniamy medianą lub średnią
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df_numeric.columns:
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    return df_numeric

