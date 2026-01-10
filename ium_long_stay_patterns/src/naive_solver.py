import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ium_long_stay_patterns.src.helper_methods import (
    prepare_booking_data,
    prepare_listing_data,
)


def naive_predict_all_zero(n_samples):
    """
    Return an array of zeros of length n_samples (predict 'not long stay' for all).
    """
    return np.zeros(n_samples, dtype=int)


def evaluate_naive_on_bookings(
    sessions_csv="data/raw/sessions.csv",
    test_size=0.2,
    random_state=42,
    verbose=True,
):
    """
    Prepare booking data, split into train/test, and evaluate the naive baseline
    that always predicts `is_long_stay = 0`.

    Returns a dict with metrics and prints a simple report.
    """
    bookings = prepare_booking_data(sessions_csv)
    bookings = bookings.dropna(subset=["is_long_stay"])
    y = bookings["is_long_stay"].astype(int).values

    # We don't need features for naive model; split indices to get stratified sample
    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_pred = naive_predict_all_zero(len(y_test))
    # For AP we need scores (confidence/probabilities). For naive all-zero predictor we
    # use zeros as scores (predict probability of class 1 = 0 for all).
    y_scores = np.zeros(len(y_test), dtype=float)

    # compute metrics (note: precision/recall for class '1' by default)
    acc = accuracy_score(y_test, y_pred)
    prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    prec_neg = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_neg = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_neg = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    # Average Precision (AP) from precision-recall curve (needs scores)
    try:
        ap = float(average_precision_score(y_test, y_scores))
    except Exception:
        ap = 0.0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results = {
        "accuracy": acc,
        "precision_pos": prec_pos,
        "recall_pos": rec_pos,
        "f1_pos": f1_pos,
        "precision_neg": prec_neg,
        "recall_neg": rec_neg,
        "f1_neg": f1_neg,
        "average_precision": ap,
        "confusion_matrix": cm,
        "n_test": len(y_test),
        "class_balance_test": {
            "n_not_long_stay": int((y_test == 0).sum()),
            "n_long_stay": int((y_test == 1).sum()),
        },
    }

    if verbose:
        print("Naive baseline: always predict is_long_stay = 0")
        print(f"Test size: {len(y_test)}")
        print(
            "Class counts (test): not_long_stay=0 ->",
            results["class_balance_test"]["n_not_long_stay"],
            ", long_stay=1 ->",
            results["class_balance_test"]["n_long_stay"],
        )
        print(f"Accuracy: {acc:.4f}")
        print(f"Average precision (AP): {ap:.4f}")
        print("Confusion matrix (rows=true, cols=pred) with labels [0,1]:")
        print(cm)
        print("\nClassification report (for positive class=1):")
        print(classification_report(y_test, y_pred, zero_division=0))

    return results


def evaluate_naive_on_listings(
    listings_csv="data/raw/listings.csv",
    sessions_csv="data/raw/sessions.csv",
    test_size=0.2,
    random_state=42,
    verbose=True,
):
    """
    Same as evaluate_naive_on_bookings but runs on listing-level DataFrame from
    `prepare_listing_data`. The target is `is_long_stay` defined in that function.
    """
    listings = prepare_listing_data(listings_csv=listings_csv, sessions_csv=sessions_csv)
    listings = listings.dropna(subset=["is_long_stay"])
    y = listings["is_long_stay"].astype(int).values

    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_pred = naive_predict_all_zero(len(y_test))
    y_scores = np.zeros(len(y_test), dtype=float)

    acc = accuracy_score(y_test, y_pred)
    try:
        ap = float(average_precision_score(y_test, y_scores))
    except Exception:
        ap = 0.0
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if verbose:
        print("Naive baseline on listings: always predict is_long_stay = 0")
        print(f"Test size: {len(y_test)}")
        print(
            "Class counts (test): not_long_stay=0 ->",
            int((y_test == 0).sum()),
            ", long_stay=1 ->",
            int((y_test == 1).sum()),
        )
        print(f"Accuracy: {acc:.4f}")
        print(f"Average precision (AP): {ap:.4f}")
        print("Confusion matrix (rows=true, cols=pred) with labels [0,1]:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "average_precision": ap,
        "confusion_matrix": cm,
        "n_test": len(y_test),
    }
