import json
import numpy as np
import pandas as pd

matches = pd.read_csv("wcData/matches_master.csv")
ratings = pd.read_csv("wcData/ratings.csv")[['team', 'rating']]
ratings_dict = dict(zip(ratings['team'], ratings['rating']))

matches['neutral'] = matches['neutral'].astype(str).str.upper() == 'TRUE'
cal = matches[(matches['neutral']) & (matches['result'] != 'D')].copy()
print(f"Neutral, decisive matches: {len(cal)} of {len(matches)} total rows")

unmatched = sorted((set(cal['team']) | set(cal['opponent'])) - set(ratings_dict))
if unmatched:
    print(f"\n{len(unmatched)} team names have no rating (rows will be dropped):")
    print(unmatched)

cal['rating'] = cal['team'].map(ratings_dict)
cal['opp_rating'] = cal['opponent'].map(ratings_dict)
before = len(cal)
cal = cal.dropna(subset=['rating', 'opp_rating'])
print(f"Dropped {before - len(cal)} rows missing a rating; {len(cal)} usable rows remain")

cal['rating_diff'] = cal['rating'] - cal['opp_rating']
cal['won'] = (cal['result'] == 'W').astype(int)


def fit_logistic(x, y, fit_intercept=True, l2=2.0, lr=0.3, iters=5000):
    x_raw = x.values.astype(float)
    y = y.values.astype(float)
    x_std = x_raw.std()
    x = x_raw / x_std

    k, b = 0.0, 0.0
    n = len(x)
    for _ in range(iters):
        z = k * x + (b if fit_intercept else 0)
        p = 1 / (1 + np.exp(-z))
        grad_k = np.dot(x, p - y) / n + (l2 / n) * k
        k -= lr * grad_k
        if fit_intercept:
            b -= lr * (np.sum(p - y) / n)
    return k / x_std, b


k_no_int, _ = fit_logistic(cal['rating_diff'], cal['won'], fit_intercept=False)
k_int, b_int = fit_logistic(cal['rating_diff'], cal['won'], fit_intercept=True)

print(f"\nFitted slope (no intercept): k = {k_no_int:.6f}")
print(f"Sanity check w/ intercept: k = {k_int:.6f}, b = {b_int:.4f}  "
      f"(b should be ~0 on neutral-only data)")

divisor = np.log(10) / k_no_int
print(f"\nOld divisor (chess default): 400")
print(f"Calibrated divisor: {divisor:.1f}")


def predicted_prob(diff, d):
    return 1 / (1 + 10 ** (-diff / d))


cal['pred_old'] = predicted_prob(cal['rating_diff'], 400)
cal['pred_new'] = predicted_prob(cal['rating_diff'], divisor)
cal['bucket'] = pd.qcut(cal['pred_new'], 10, duplicates='drop')

report = cal.groupby('bucket').agg(
    n=('won', 'size'),
    actual_win_rate=('won', 'mean'),
    avg_pred_new=('pred_new', 'mean'),
    avg_pred_old=('pred_old', 'mean'),
).round(3)
print("\nCalibration check (predicted vs actual, by bucket):")
print(report.to_string())

with open("wcData/calibration.json", "w") as f:
    json.dump({"divisor": divisor, "n_matches": len(cal)}, f, indent=2)
print("\nSaved -> wcData/calibration.json")