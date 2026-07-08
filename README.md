#  World Cup Knockout Predictor & Monte Carlo Simulator

This repository contains an end-to-end predictive analytics framework designed for the World Cup knockout stages. Rather than using static brackets, the model implements a custom Team Strength Rating (TSR) engine alongside a 10,000-iteration Monte Carlo simulation to project the probability of each team advancing through every tournament round.---

##  Project Overview

This architecture replaces traditional tournament brackets with a dynamic, data-driven assessment of team quality. By consolidating match performance data, historical strength indicators, and real-time betting market sentiment, the pipeline establishes a standardized power rating for all 32 competing nations to simulate potential tournament paths.
### Core Features
* ****Multi-Factor Rating Engine:** Integrates forward-looking statistics (xG differential), lagging metrics (Points Per Game), schedule context (Strength of Schedule), and consensus market evaluation (implied betting probabilities).
* **Monte Carlo Simulation Engine:** Executes 10,000+ single-elimination tournament runs powered by a calibrated logistic distribution function.
* **Interactive Dashboard:** A front-end interface featuring a dynamic bracket tree, a sorting-friendly matrix view, and modular team profiles for deep-dive analysis.

---

##  Rating Methodology & Normalization

To ensure comparability between metrics operating on entirely different scales (such as Elo ratings versus points-per-game), the pipeline applies Z-score normalization to every baseline feature. This transforms each variable to represent its distance from the tournament mean in units of standard deviation.

$$\ z = \frac{x - \mu}{\sigma} \$$

### Default Feature Weights
The model aggregates features using a weighted composite index tailored for high predictive accuracy in international football:

| Metric Group | Specific Feature | Default Weight | Rationale |
| :--- | :--- | :--- | :--- |
| **Performance** | xG Differential per match | `0.25` | Strongest on-pitch indicator of future performance |
| **Market** | Normalized Implied Probability | `0.20` | Captures real-time squad depth, injuries, and public consensus |
| **Contextual** | Strength of Schedule (SOS) | `0.15` | Penalizes stats inflated against weaker opposition |
| **Historical** | World Elo Rating | `0.15` | Time-tested, chess-based metric for international strength |
| **Momentum** | Exponentially Decayed Form | `0.15` | Weighting matches using a time-decay factor ($\alpha = 0.85$) |
| **Traditional** | Goal Difference per match | `0.10` | Standard margin of victory tracking |
| **Seeding** | Inverse FIFA Ranking | `0.05` | Useful baseline, though slower to react to recent form |

### Final Scaling
To make the output intuitive, the raw composite score is mapped to a standard Elo scale with a designated mean of 1500 and a standard deviation of 300. This is similar to other rating sports/games like chess. 

$$\ \text{Final Rating} = 1500 + 300 \times \sum (w_i \cdot z_i) \$$

---

##  Data Pipeline & Database Schema

The pipeline centers around a symmetrical **Master Match Results Table** where every game is mirrored from the perspective of both competing teams. This architecture allows fast, grouped aggregate operations (`df.groupby('team')`).

### Database Schema Layout
* `match_id` (int64): Unique game identifier.
* `date` (datetime64): Date of the fixture for time-decay form tracking.
* `team` / `opponent` (str): Participating nations.
* `goals_for` / `goals_against` (int32): On-pitch scorelines.
* `xg_for` / `xg_against` (float64): Expected Goals variables.

---

##  Simulation Engine

Individual match outcomes are determined using a standard logistic curve based on the rating margin between the two teams.

$$\ P(\text{Team A}) = \frac{1}{1 + 10^{-\frac{\Delta R}{400}}} \$$

Where $\Delta R$ is the difference between `Rating_A` and `Rating_B`. 

The simulation loops through the single-elimination structure, tracking winners through a cascading array. The algorithm records the exact frequency with which each team reaches the Round of 16, Quarterfinals, Semifinals, Finals, and wins the championship across all 10,000 iterations.

---

##  UI & Dashboard Architecture

The frontend is split into intuitive user views designed for clarity at a glance:

1. **The Interactive Bracket Tree:** A structural visualization of the tournament tree using color gradients to communicate survival probabilities. Users can manually select winners to force "What-If" scenarios, triggering an immediate recalculation of all remaining paths.
2. **The Tournament Matrix:** A tabular leaderboard showing all 32 teams alongside their precise mathematical likelihood of advancing past each phase.
3. **Team Deep-Dive Panel:** A sliding drawer component featuring a **5-axis Radar Chart** (Attack, Defense, Form, Schedule, Market) and a predictive projection of their most likely route to the finals.

---

##  Getting Started

### Prerequisites
```bash
pip install pandas numpy plotly streamlit
