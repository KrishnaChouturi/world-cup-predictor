import json
import os
import pandas as pd
import numpy as np

ratings = pd.read_csv("wcData/ratings.csv")[['team', 'rating']]


def load_divisor(default=400):
    path = "wcData/calibration.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["divisor"]
    print("WARNING: wcData/calibration.json not found, using uncalibrated default 400")
    return default


DIVISOR = load_divisor()


def win_probability(rating_a, rating_b):
    diff = rating_a - rating_b
    return 1 / (1 + 10 ** (-diff / DIVISOR))


def simulate_match(team_a, team_b, ratings_dict):
    r_a = ratings_dict[team_a]
    r_b = ratings_dict[team_b]
    p_a = win_probability(r_a, r_b)
    return team_a if np.random.random() < p_a else team_b


def simulate_bracket(teams, ratings_dict):
    current_round = teams[:]
    results = {team: {'R32': 0, 'R16': 0, 'QF': 0, 'SF': 0, 'F': 0, 'W': 0} for team in teams}
    round_names = ['R16', 'QF', 'SF', 'F', 'W']
    round_idx = 0

    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_match(current_round[i], current_round[i+1], ratings_dict)
            next_round.append(winner)
            results[winner][round_names[round_idx]] += 1
        current_round = next_round
        round_idx += 1

    return results


def monte_carlo(teams, ratings_dict, n=10000):
    totals = {team: {'R16': 0, 'QF': 0, 'SF': 0, 'F': 0, 'W': 0} for team in teams}

    for _ in range(n):
        result = simulate_bracket(teams, ratings_dict)
        for team in teams:
            for stage in ['R16', 'QF', 'SF', 'F', 'W']:
                totals[team][stage] += result[team][stage]

    probs = {}
    for team in teams:
        probs[team] = {stage: round(totals[team][stage] / n, 4) for stage in ['R16', 'QF', 'SF', 'F', 'W']}

    return probs


# --- Input your Round of 32 bracket here ---
bracket = [
    'Argentina', 'Australia',
    'France', 'Morocco',
    'Spain', 'Iran',
    'England', 'Senegal',
    'Germany', 'Japan',
    'Portugal', 'South Korea',
    'Netherlands', 'Mexico',
    'Brazil', 'Switzerland',
    'Belgium', 'Algeria',
    'Italy', 'Nigeria',
    'Colombia', 'Canada',
    'Uruguay', 'Denmark',
    'Croatia', 'Turkey',
    'United States', 'Poland',
    'Ivory Coast', 'Ecuador',
    'Hungary', 'Norway',
]

ratings_dict = dict(zip(ratings['team'], ratings['rating']))

missing = [t for t in bracket if t not in ratings_dict]
if missing:
    print("WARNING - missing ratings for:", missing)
else:
    print("All teams have ratings. Running simulation...\n")
    probs = monte_carlo(bracket, ratings_dict, n=10000)

    results_df = pd.DataFrame(probs).T
    results_df.index.name = 'team'
    results_df = results_df.sort_values('W', ascending=False)

    print(results_df.to_string())

    results_df.to_csv("wcData/simulation_results.csv")
    print("\nSaved to wcData/simulation_results.csv")