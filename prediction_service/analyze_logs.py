"""Calculate business and analytical metrics from A/B test logs.

Reads logs_ab.txt (JSON lines format) and computes various metrics for comparing
the naive and binary models.

Usage: poetry run python prediction_service/calculate_metrics.py
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd
from scipy import stats
import numpy as np

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "logs_ab.txt"


def load_logs(path: Path):
    """Load JSON lines log file into a list of dictionaries."""
    logs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def parse_prediction(pred):
    """Extract prediction value from various formats."""
    if pred is None:
        return None
    if isinstance(pred, list):
        return pred[0] if len(pred) > 0 else None
    return pred


def calculate_metrics(logs):
    """Calculate comprehensive metrics from A/B test logs."""

    model_data = defaultdict(lambda: {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'predictions': [],
        'timestamps': [],
        'response_times': [],
        'positive_predictions': 0,
        'negative_predictions': 0,
    })

    for log in logs:
        model = log.get('model') or log.get('model_chosen')
        if not model:
            continue

        data = model_data[model]
        data['total_requests'] += 1
        data['timestamps'].append(log.get('timestamp'))

        status = log.get('status_code')
        if status == 200:
            data['successful_requests'] += 1
            pred = parse_prediction(log.get('prediction'))
            if pred is not None:
                data['predictions'].append(pred)
                if pred > 0:
                    data['positive_predictions'] += 1
                else:
                    data['negative_predictions'] += 1
        else:
            data['failed_requests'] += 1

    results = {}

    for model, data in model_data.items():
        total = data['total_requests']
        successful = data['successful_requests']

        metrics = {
            # Volume metrics
            'total_requests': total,
            'successful_requests': successful,
            'failed_requests': data['failed_requests'],

            # Reliability metrics
            'success_rate': successful / total if total > 0 else 0,
            'failure_rate': data['failed_requests'] / total if total > 0 else 0,

            # Prediction distribution
            'positive_predictions': data['positive_predictions'],
            'negative_predictions': data['negative_predictions'],
            'positive_rate': data['positive_predictions'] / successful if successful > 0 else 0,

            # Business metrics
            'conversion_rate': data['positive_predictions'] / total if total > 0 else 0,
        }

        # Statistical metrics
        if len(data['predictions']) > 0:
            preds = np.array(data['predictions'])
            metrics['mean_prediction'] = float(np.mean(preds))
            metrics['std_prediction'] = float(np.std(preds))
            metrics['median_prediction'] = float(np.median(preds))

        results[model] = metrics

    return results, model_data


def compare_models(results, model_data):
    """Compare two models statistically."""

    if 'naive' not in results or 'binary' not in results:
        print("Warning: Both 'naive' and 'binary' models required for comparison")
        return {}

    naive_preds = np.array(model_data['naive']['predictions'])
    binary_preds = np.array(model_data['binary']['predictions'])

    comparison = {}

    naive_rate = results['naive']['positive_rate']
    binary_rate = results['binary']['positive_rate']

    comparison['conversion_rate_diff'] = binary_rate - naive_rate
    comparison['conversion_rate_lift'] = ((binary_rate - naive_rate) / naive_rate * 100) if naive_rate > 0 else 0

    naive_success = results['naive']['success_rate']
    binary_success = results['binary']['success_rate']

    comparison['success_rate_diff'] = binary_success - naive_success

    n1 = results['naive']['successful_requests']
    n2 = results['binary']['successful_requests']
    p1 = results['naive']['positive_predictions']
    p2 = results['binary']['positive_predictions']

    if n1 > 0 and n2 > 0:
        p_pooled = (p1 + p2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se > 0:
            z_score = (binary_rate - naive_rate) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            comparison['z_score'] = z_score
            comparison['p_value'] = p_value
            comparison['statistically_significant'] = p_value < 0.05

    if n1 > 0 and n2 > 0:
        contingency_table = np.array([
            [p1, n1 - p1],
            [p2, n2 - p2]
        ])
        chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
        comparison['chi2_statistic'] = chi2
        comparison['chi2_p_value'] = p_chi

    return comparison


def print_report(results, comparison):
    """Print a formatted metrics report."""

    print("="*80)
    print("A/B TEST METRICS REPORT")
    print("="*80)
    print()

    for model, metrics in results.items():
        print(f"{'='*80}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*80}")
        print()

        print("VOLUME METRICS:")
        print(f"  Total Requests:      {metrics['total_requests']:,}")
        print(f"  Successful:          {metrics['successful_requests']:,}")
        print(f"  Failed:              {metrics['failed_requests']:,}")
        print()

        print("RELIABILITY METRICS:")
        print(f"  Success Rate:        {metrics['success_rate']:.2%}")
        print(f"  Failure Rate:        {metrics['failure_rate']:.2%}")
        print()

        print("PREDICTION METRICS:")
        print(f"  Longstay Predictions: {metrics['positive_predictions']:,}")
        print(f"  Shortstay Predictions: {metrics['negative_predictions']:,}")
        print(f"  Longstay Rate:        {metrics['positive_rate']:.2%}")
        print()

        print("BUSINESS METRICS:")
        print(f"  Conversion Rate:     {metrics['conversion_rate']:.2%}")
        print()

        if 'mean_prediction' in metrics:
            print("STATISTICAL METRICS:")
            print(f"  Mean Prediction:     {metrics['mean_prediction']:.4f}")
            print(f"  Std Prediction:      {metrics['std_prediction']:.4f}")
            print(f"  Median Prediction:   {metrics['median_prediction']:.4f}")
            print()

    if comparison:
        print(f"{'='*80}")
        print("MODEL COMPARISON (Binary vs Naive)")
        print(f"{'='*80}")
        print()

        print("CONVERSION RATE:")
        print(f"  Difference:          {comparison['conversion_rate_diff']:+.2%}")
        print(f"  Lift:                {comparison['conversion_rate_lift']:+.2f}%")
        print()

        print("SUCCESS RATE:")
        print(f"  Difference:          {comparison['success_rate_diff']:+.2%}")
        print()

        if 'p_value' in comparison:
            print("STATISTICAL SIGNIFICANCE:")
            print(f"  Z-Score:             {comparison['z_score']:.4f}")
            print(f"  P-Value:             {comparison['p_value']:.4f}")
            print(f"  Significant (α=0.05): {comparison['statistically_significant']}")
            print()

        if 'chi2_p_value' in comparison:
            print("CHI-SQUARE TEST:")
            print(f"  Chi² Statistic:      {comparison['chi2_statistic']:.4f}")
            print(f"  P-Value:             {comparison['chi2_p_value']:.4f}")
            print()

    print("="*80)


def save_metrics_to_csv(results, comparison, output_path="metrics_summary.csv"):
    """Save metrics to CSV for further analysis."""

    rows = []
    for model, metrics in results.items():
        row = {'model': model}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nMetrics saved to {output_path}")

    if comparison:
        comp_df = pd.DataFrame([comparison])
        comp_path = output_path.replace('.csv', '_comparison.csv')
        comp_df.to_csv(comp_path, index=False)
        print(f"Comparison metrics saved to {comp_path}")


def main():
    if not LOG_PATH.exists():
        print(f"Error: Log file not found at {LOG_PATH}")
        return

    print(f"Loading logs from {LOG_PATH}...")
    logs = load_logs(LOG_PATH)
    print(f"Loaded {len(logs)} log entries")
    print()

    results, model_data = calculate_metrics(logs)
    comparison = compare_models(results, model_data)

    print_report(results, comparison)
    save_metrics_to_csv(results, comparison)


if __name__ == "__main__":
    main()
