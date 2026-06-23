import json
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

ratings_df = pd.read_csv("wcData/ratings.csv")[['team', 'rating']]
ratings_dict = dict(zip(ratings_df['team'], ratings_df['rating']))


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


def simulate_match(team_a, team_b):
    r_a = ratings_dict.get(team_a, 1500)
    r_b = ratings_dict.get(team_b, 1500)
    p_a = win_probability(r_a, r_b)
    return team_a if np.random.random() < p_a else team_b


def simulate_bracket(teams):
    current_round = teams[:]
    round_names = ['R16', 'QF', 'SF', 'F', 'W']
    results = {team: {r: 0 for r in round_names} for team in teams}
    round_idx = 0

    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_match(current_round[i], current_round[i+1])
            next_round.append(winner)
            results[winner][round_names[round_idx]] += 1
        current_round = next_round
        round_idx += 1

    return results


def monte_carlo(teams, n=10000):
    totals = {team: {'R16': 0, 'QF': 0, 'SF': 0, 'F': 0, 'W': 0} for team in teams}
    for _ in range(n):
        result = simulate_bracket(teams)
        for team in teams:
            for stage in ['R16', 'QF', 'SF', 'F', 'W']:
                totals[team][stage] += result[team][stage]
    probs = {}
    for team in teams:
        probs[team] = {
            stage: round(totals[team][stage] / n, 4)
            for stage in ['R16', 'QF', 'SF', 'F', 'W']
        }
    return probs


@app.route('/')
def index():
    teams = sorted(ratings_dict.keys())
    return render_template('index.html', teams=teams)


@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    bracket = data.get('bracket', [])

    if len(bracket) != 32:
        return jsonify({'error': f'Expected 32 teams, got {len(bracket)}'}), 400

    missing = [t for t in bracket if t not in ratings_dict]
    if missing:
        return jsonify({'error': f'Missing ratings for: {missing}'}), 400

    probs = monte_carlo(bracket)
    return jsonify(probs)


if __name__ == '__main__':
    app.run(debug=True)