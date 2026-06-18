import pandas as pd

master = pd.read_csv("wcData/master_ratings.csv")
df = master.dropna(subset=['fifa_rank']).copy()

# Z-score normalization
def zscore(series):
    return (series - series.mean()) / series.std()

df['ppg_norm']         = zscore(df['ppg'])
df['gd_norm']          = zscore(df['gd_per_game'])
df['recent_form_norm'] = zscore(df['recent_form'])
df['sos_norm']         = zscore(df['sos'])
df['elo_norm']         = zscore(df['elo'])
df['wc_prob_norm']     = zscore(df['wc_prob'])
df['fifa_norm']        = zscore(df['fifa_score'])

# Weights
w_ppg         = 0.15
w_gd          = 0.15
w_recent_form = 0.20
w_sos         = 0.10
w_elo         = 0.20
w_wc_prob     = 0.10
w_fifa        = 0.10


df['rating_raw'] = (
    w_ppg         * df['ppg_norm'] +
    w_gd          * df['gd_norm'] +
    w_recent_form * df['recent_form_norm'] +
    w_sos         * df['sos_norm'] +
    w_elo         * df['elo_norm'] +
    w_wc_prob     * df['wc_prob_norm'] +
    w_fifa        * df['fifa_norm']
)

df['rating'] = 1500 + 300 * df['rating_raw']

print("Top 30 teams by rating:")
print(
    df.sort_values('rating', ascending=False)
    [['team', 'rating', 'ppg', 'gd_per_game', 'recent_form', 'sos', 'elo', 'wc_prob']]
    .head(30)
    .to_string(index=False)
)

df.to_csv("wcData/ratings.csv", index=False)
print("\nSaved to wcData/ratings.csv")