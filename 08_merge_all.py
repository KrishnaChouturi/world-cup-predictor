import pandas as pd

combined = pd.read_csv("wcData/combined_metrics.csv")
external = pd.read_csv("wcData/external_data.csv")

master = combined.merge(external, on='team', how='left')

print("Master shape:", master.shape)
print("\nColumns:", master.columns.tolist())

contenders = master.dropna(subset=['fifa_rank'])
print(f"\nTeams with full data: {len(contenders)}")
print("\nTop 20 by PPG:")
print(
    contenders.sort_values('ppg', ascending=False)
    [['team', 'games', 'ppg', 'gd_per_game', 'recent_form', 'sos', 'elo', 'wc_prob']]
    .head(20)
    .to_string(index=False)
)

master.to_csv("wcData/master_ratings.csv", index=False)
print("\nSaved to wcData/master_ratings.csv")