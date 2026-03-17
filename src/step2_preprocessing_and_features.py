"""
Bước 1b + Bước 2: Preprocessing & Feature Engineering
- Clean data, filter relevant matches
- Compute Elo ratings for all teams
- Compute Poisson attack/defense strength
- Compute form (last 5 games), head-to-head, goal difference
- Output: final feature matrix ready for XGBoost (Bước 3)
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# BƯỚC 1b: PREPROCESSING
# ============================================================

def load_and_clean_data():
    """Load raw data and perform basic cleaning."""
    print("=== Bước 1b: Preprocessing ===")

    df = pd.read_csv(os.path.join(RAW_DIR, 'results.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Filter: only matches from 2000 onwards (more relevant for modern football)
    df = df[df['date'] >= '2000-01-01'].copy()

    # Create result column (target variable for XGBoost)
    # 2 = Home Win, 1 = Draw, 0 = Away Win
    df['result'] = np.where(
        df['home_score'] > df['away_score'], 2,
        np.where(df['home_score'] == df['away_score'], 1, 0)
    )

    # Convert neutral to int
    df['is_neutral'] = df['neutral'].astype(int)

    # Tournament weight: competitive matches matter more for Elo
    competitive_tournaments = [
        'FIFA World Cup', 'FIFA World Cup qualification',
        'UEFA Euro', 'UEFA Euro qualification',
        'Copa América', 'Copa América qualification',
        'African Cup of Nations', 'African Cup of Nations qualification',
        'AFC Asian Cup', 'AFC Asian Cup qualification',
        'CONCACAF Gold Cup', 'UEFA Nations League',
        'Confederations Cup', 'CONMEBOL–UEFA Cup of Champions',
    ]
    df['is_competitive'] = df['tournament'].isin(competitive_tournaments).astype(int)

    print(f"  Total matches (2000+): {len(df)}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Teams: {df['home_team'].nunique()}")
    print(f"  Result distribution: {df['result'].value_counts().to_dict()}")

    return df


# ============================================================
# BƯỚC 2a: ELO RATINGS
# ============================================================

def compute_elo_ratings(df, k_base=30, home_advantage=100):
    """
    Compute Elo ratings for all teams across all matches.
    Returns df with elo_home, elo_away, elo_diff columns.
    """
    print("\n=== Bước 2a: Elo Ratings ===")

    elo = defaultdict(lambda: 1500.0)  # Default Elo = 1500
    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        neutral = row['is_neutral']

        # Current Elo before match
        home_elo = elo[home] + (0 if neutral else home_advantage)
        away_elo = elo[away]

        elo_home_list.append(elo[home])
        elo_away_list.append(elo[away])

        # Expected scores
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1 - exp_home

        # Actual scores
        if row['home_score'] > row['away_score']:
            actual_home, actual_away = 1, 0
        elif row['home_score'] == row['away_score']:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0, 1

        # K-factor adjustment: competitive matches have higher K
        k = k_base * (1.5 if row['is_competitive'] else 1.0)

        # Goal difference multiplier
        goal_diff = abs(row['home_score'] - row['away_score'])
        if goal_diff <= 1:
            gd_mult = 1
        elif goal_diff == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11 + goal_diff) / 8

        k = k * gd_mult

        # Update Elo
        elo[home] += k * (actual_home - exp_home)
        elo[away] += k * (actual_away - exp_away)

    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away']

    # Show top 20 teams by current Elo
    top_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:20]
    print("  Top 20 teams by Elo:")
    for i, (team, rating) in enumerate(top_teams, 1):
        print(f"    {i:2d}. {team:25s} {rating:.0f}")

    return df, dict(elo)


# ============================================================
# BƯỚC 2a (cont): RANK DIFF (from FIFA World Ranking)
# ============================================================

# Mapping: FIFA ranking name -> results.csv name
FIFA_TO_RESULTS_NAME = {
    'USA': 'United States',
    'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea',
    'IR Iran': 'Iran',
    'Côte d\'Ivoire': 'Ivory Coast',
    'Congo DR': 'DR Congo',
    'Cabo Verde': 'Cape Verde',
    'Czechia': 'Czech Republic',
    'Chinese Taipei': 'Taiwan',
    'Brunei Darussalam': 'Brunei',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'St Kitts and Nevis': 'Saint Kitts and Nevis',
    'St Lucia': 'Saint Lucia',
    'St Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'The Gambia': 'Gambia',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Curacao': 'Curaçao',
    'Serbia and Montenegro': 'Serbia and Montenegro',
    'Zaire': 'Zaire',
}


def load_fifa_ranking():
    """Load and prepare FIFA World Ranking data."""
    # Use the most recent/complete file
    ranking_files = sorted([
        f for f in os.listdir(RAW_DIR) if f.startswith('fifa_ranking')
    ])
    # Concatenate all ranking files, dedup by (country, date)
    dfs = []
    for f in ranking_files:
        rdf = pd.read_csv(os.path.join(RAW_DIR, f))
        dfs.append(rdf)

    ranking = pd.concat(dfs, ignore_index=True)
    ranking = ranking.dropna(subset=['rank', 'country_full', 'rank_date'])
    ranking['rank'] = ranking['rank'].astype(int)
    ranking['total_points'] = ranking['total_points'].astype(float)
    ranking['rank_date'] = pd.to_datetime(ranking['rank_date'])

    # Apply name mapping
    ranking['team'] = ranking['country_full'].replace(FIFA_TO_RESULTS_NAME)

    # Deduplicate: keep one rank per team per date
    ranking = ranking.sort_values('rank_date').drop_duplicates(
        subset=['team', 'rank_date'], keep='last'
    )

    print(f"  FIFA Ranking loaded: {len(ranking)} entries, "
          f"{ranking['team'].nunique()} teams, "
          f"{ranking['rank_date'].min().date()} -> {ranking['rank_date'].max().date()}")

    return ranking[['team', 'rank', 'total_points', 'rank_date']].copy()


def compute_rank_diff(df):
    """
    Compute rank_diff and points_diff using real FIFA World Ranking data.
    For each match, find the most recent ranking before the match date.
    """
    print("\n=== Computing Rank Diff (FIFA World Ranking) ===")

    ranking = load_fifa_ranking()

    # Build lookup: for each team, sorted list of (date, rank, points)
    team_ranking = defaultdict(list)
    for _, row in ranking.iterrows():
        team_ranking[row['team']].append(
            (row['rank_date'], row['rank'], row['total_points'])
        )
    # Sort by date
    for team in team_ranking:
        team_ranking[team].sort(key=lambda x: x[0])

    def get_rank_at_date(team, match_date):
        """Get most recent rank before match_date using binary search."""
        history = team_ranking.get(team)
        if not history:
            return None, None
        # Binary search for latest date <= match_date
        lo, hi = 0, len(history) - 1
        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if history[mid][0] <= match_date:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if result is not None:
            return history[result][1], history[result][2]  # rank, points
        return None, None

    rank_home_list = []
    rank_away_list = []
    points_home_list = []
    points_away_list = []

    for _, row in df.iterrows():
        match_date = row['date']
        rh, ph = get_rank_at_date(row['home_team'], match_date)
        ra, pa = get_rank_at_date(row['away_team'], match_date)
        rank_home_list.append(rh)
        rank_away_list.append(ra)
        points_home_list.append(ph)
        points_away_list.append(pa)

    df['rank_home'] = rank_home_list
    df['rank_away'] = rank_away_list
    df['points_home'] = points_home_list
    df['points_away'] = points_away_list

    # Fill missing ranks with median (for teams not in FIFA ranking)
    median_rank = df['rank_home'].median()
    median_points = df['points_home'].median()
    for col in ['rank_home', 'rank_away']:
        df[col] = df[col].fillna(median_rank)
    for col in ['points_home', 'points_away']:
        df[col] = df[col].fillna(median_points)

    # rank_diff: lower rank = better, so away - home (positive = home is better)
    df['rank_diff'] = df['rank_away'] - df['rank_home']
    df['points_diff'] = df['points_home'] - df['points_away']

    matched = df['rank_home'].notna().sum()
    print(f"  Matched {matched}/{len(df)} matches with FIFA ranking")
    print(f"  rank_diff range: [{df['rank_diff'].min():.0f}, {df['rank_diff'].max():.0f}]")
    print(f"  points_diff range: [{df['points_diff'].min():.0f}, {df['points_diff'].max():.0f}]")

    return df


# ============================================================
# BƯỚC 2b: POISSON ATTACK/DEFENSE STRENGTH
# ============================================================

def compute_poisson_strength(df, window_matches=30):
    """
    Compute attack and defense strength for home and away teams
    based on Poisson regression logic.

    attack_strength = team's avg goals scored / league avg goals
    defense_strength = team's avg goals conceded / league avg goals conceded
    """
    print(f"\n=== Bước 2b: Poisson Attack/Defense Strength (window={window_matches}) ===")

    # We compute rolling stats per team
    teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    # Track each team's recent goals scored and conceded
    team_goals_scored = defaultdict(list)    # goals scored in recent matches
    team_goals_conceded = defaultdict(list)  # goals conceded in recent matches

    home_attack_list = []
    home_defense_list = []
    away_attack_list = []
    away_defense_list = []
    expected_goals_home_list = []
    expected_goals_away_list = []
    expected_goals_diff_list = []

    # Global average (running)
    all_goals = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Global average goals per match (so far)
        if len(all_goals) >= 20:
            avg_goals = np.mean(all_goals[-500:])  # recent 500 matches avg
        else:
            avg_goals = 1.4  # default

        # Home team stats (from their last N matches)
        h_scored = team_goals_scored[home][-window_matches:]
        h_conceded = team_goals_conceded[home][-window_matches:]

        # Away team stats
        a_scored = team_goals_scored[away][-window_matches:]
        a_conceded = team_goals_conceded[away][-window_matches:]

        # Compute strengths
        if len(h_scored) >= 5 and len(a_scored) >= 5:
            home_attack = np.mean(h_scored) / max(avg_goals, 0.5)
            home_defense = np.mean(h_conceded) / max(avg_goals, 0.5)
            away_attack = np.mean(a_scored) / max(avg_goals, 0.5)
            away_defense = np.mean(a_conceded) / max(avg_goals, 0.5)

            # Expected goals (Poisson lambda)
            exp_goals_home = home_attack * away_defense * avg_goals
            exp_goals_away = away_attack * home_defense * avg_goals
        else:
            home_attack = 1.0
            home_defense = 1.0
            away_attack = 1.0
            away_defense = 1.0
            exp_goals_home = avg_goals
            exp_goals_away = avg_goals

        home_attack_list.append(home_attack)
        home_defense_list.append(home_defense)
        away_attack_list.append(away_attack)
        away_defense_list.append(away_defense)
        expected_goals_home_list.append(exp_goals_home)
        expected_goals_away_list.append(exp_goals_away)
        expected_goals_diff_list.append(exp_goals_home - exp_goals_away)

        # Update team histories AFTER using them (no data leakage)
        team_goals_scored[home].append(row['home_score'])
        team_goals_conceded[home].append(row['away_score'])
        team_goals_scored[away].append(row['away_score'])
        team_goals_conceded[away].append(row['home_score'])
        all_goals.append(row['home_score'])
        all_goals.append(row['away_score'])

    df['home_attack_strength'] = home_attack_list
    df['home_defense_strength'] = home_defense_list
    df['away_attack_strength'] = away_attack_list
    df['away_defense_strength'] = away_defense_list
    df['expected_goals_home'] = expected_goals_home_list
    df['expected_goals_away'] = expected_goals_away_list
    df['expected_goals_diff'] = expected_goals_diff_list

    print("  Done. Added: home/away_attack_strength, home/away_defense_strength,")
    print("               expected_goals_home/away/diff")
    return df


# ============================================================
# BƯỚC 2c: FORM, HEAD-TO-HEAD, GOAL DIFFERENCE
# ============================================================

def compute_form_features(df, n_games=5):
    """
    Compute form features for each team based on last N games.
    - Form score: Win=3, Draw=1, Loss=0 (normalized to 0-1)
    - Goal difference in last N games
    """
    print(f"\n=== Bước 2c: Form Features (last {n_games} games) ===")

    team_results = defaultdict(list)     # list of (points, goal_diff)
    home_form_list = []
    away_form_list = []
    home_gd_list = []
    away_gd_list = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Home team form
        recent_home = team_results[home][-n_games:]
        if len(recent_home) >= 3:
            home_form = np.mean([r[0] for r in recent_home]) / 3.0  # normalize to 0-1
            home_gd = np.mean([r[1] for r in recent_home])
        else:
            home_form = 0.5
            home_gd = 0.0

        # Away team form
        recent_away = team_results[away][-n_games:]
        if len(recent_away) >= 3:
            away_form = np.mean([r[0] for r in recent_away]) / 3.0
            away_gd = np.mean([r[1] for r in recent_away])
        else:
            away_form = 0.5
            away_gd = 0.0

        home_form_list.append(home_form)
        away_form_list.append(away_form)
        home_gd_list.append(home_gd)
        away_gd_list.append(away_gd)

        # Update histories AFTER (no leakage)
        h_score = row['home_score']
        a_score = row['away_score']

        if h_score > a_score:
            team_results[home].append((3, h_score - a_score))
            team_results[away].append((0, a_score - h_score))
        elif h_score == a_score:
            team_results[home].append((1, 0))
            team_results[away].append((1, 0))
        else:
            team_results[home].append((0, h_score - a_score))
            team_results[away].append((3, a_score - h_score))

    df['home_form'] = home_form_list
    df['away_form'] = away_form_list
    df['goal_diff_home_5'] = home_gd_list
    df['goal_diff_away_5'] = away_gd_list

    print("  Done. Added: home_form, away_form, goal_diff_home/away_5")
    return df


def compute_h2h_features(df):
    """
    Compute head-to-head win rate for home team vs away team.
    """
    print("\n=== Computing Head-to-Head Features ===")

    h2h_record = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'total': 0})
    h2h_list = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        key = tuple(sorted([home, away]))

        record = h2h_record[key]
        if record['total'] >= 3:
            # Win rate from home team's perspective
            if home < away:
                h2h_wr = record['wins'] / record['total']
            else:
                h2h_wr = record['losses'] / record['total']
        else:
            h2h_wr = 0.5  # default when insufficient history

        h2h_list.append(h2h_wr)

        # Update h2h AFTER
        if row['home_score'] > row['away_score']:
            if home < away:
                record['wins'] += 1
            else:
                record['losses'] += 1
        elif row['home_score'] == row['away_score']:
            record['draws'] += 1
        else:
            if home < away:
                record['losses'] += 1
            else:
                record['wins'] += 1
        record['total'] += 1

    df['h2h_win_rate'] = h2h_list

    print("  Done. Added: h2h_win_rate")
    return df


# ============================================================
# FINAL: ASSEMBLE FEATURE MATRIX
# ============================================================

def assemble_features(df):
    """Select final feature columns and save."""
    print("\n=== Assembling Final Feature Matrix ===")

    feature_cols = [
        # Elo
        'elo_home', 'elo_away', 'elo_diff',
        # FIFA Ranking
        'rank_home', 'rank_away', 'rank_diff',
        'points_home', 'points_away', 'points_diff',
        # Poisson-based
        'home_attack_strength', 'home_defense_strength',
        'away_attack_strength', 'away_defense_strength',
        'expected_goals_home', 'expected_goals_away', 'expected_goals_diff',
        # Form
        'home_form', 'away_form',
        'goal_diff_home_5', 'goal_diff_away_5',
        # Head-to-head
        'h2h_win_rate',
        # Context
        'is_neutral', 'is_competitive',
    ]

    metadata_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'tournament', 'result']

    final_df = df[metadata_cols + feature_cols].copy()

    # Remove early matches where features are unreliable (cold start)
    # Keep matches from 2005+ (giving 5 years of history for Elo warmup)
    final_df = final_df[final_df['date'] >= '2005-01-01'].copy()

    print(f"  Final dataset shape: {final_df.shape}")
    print(f"  Date range: {final_df['date'].min().date()} -> {final_df['date'].max().date()}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Missing values: {final_df[feature_cols].isnull().sum().sum()}")

    # Save
    output_path = os.path.join(PROCESSED_DIR, 'features_matrix.csv')
    final_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # Print feature summary
    print(f"\n  Feature Statistics:")
    print(final_df[feature_cols].describe().round(3).to_string())

    # Print result distribution
    print(f"\n  Result distribution:")
    result_map = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
    for val, label in result_map.items():
        count = (final_df['result'] == val).sum()
        pct = count / len(final_df) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    return final_df


# ============================================================
# MAIN
# ============================================================

def main():
    # Bước 1b: Load & Clean
    df = load_and_clean_data()

    # Bước 2a: Elo Ratings
    df, elo_dict = compute_elo_ratings(df)

    # Bước 2a (cont): Rank diff
    df = compute_rank_diff(df)

    # Bước 2b: Poisson Strength
    df = compute_poisson_strength(df)

    # Bước 2c: Form + H2H
    df = compute_form_features(df)
    df = compute_h2h_features(df)

    # Assemble final matrix
    final_df = assemble_features(df)

    # Save Elo dict for later use (Bước 3/4)
    import json
    elo_path = os.path.join(PROCESSED_DIR, 'elo_ratings.json')
    with open(elo_path, 'w') as f:
        json.dump(elo_dict, f, indent=2)
    print(f"\n  Elo ratings saved to: {elo_path}")

    print("\n=== Pipeline Complete! Ready for Bước 3: XGBoost ===")
    return final_df


if __name__ == '__main__':
    main()
