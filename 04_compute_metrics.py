import pandas as pd
import numpy as np

matches = pd.read_csv("wcData/matches_master.csv")
matches['date'] = pd.to_datetime(matches['date'])

# PPG
summary = matches.groupby('team').agg(
    games=('points', 'count'),
    total_points=('points', 'sum'),
    total_gd=('goal_diff', 'sum')
).reset_index()

summary['ppg'] = summary['total_points'] / summary['games']
summary['gd_per_game'] = summary['total_gd'] / summary['games']

# Recency Weighted
def recent_form(group):
    last8 = group.sort_values('date').tail(8)
    n = len(last8)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return float(np.dot(last8['points'].values, weights))

form = matches.groupby('team').apply(recent_form).reset_index()
form.columns = ['team', 'recent_form']

# Merge
metrics = summary.merge(form, on='team')

print("Metrics shape:", metrics.shape)
print("\nTop 15 by PPG:")
print(metrics.sort_values('ppg', ascending=False)[['team', 'games', 'ppg', 'gd_per_game', 'recent_form']].head(15))

metrics.to_csv("wcData/metrics.csv", index=False)
print("\nSaved to wcData/metrics.csv")