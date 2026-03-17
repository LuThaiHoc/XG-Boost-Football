"""
Web app: Visualize XGBoost model results & Match Predictor
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '..')
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

app = Flask(__name__)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

def load_all():
    """Load model, metadata, and features matrix."""
    model = joblib.load(os.path.join(PROJECT_DIR, 'models', 'xgboost_model.joblib'))
    with open(os.path.join(PROJECT_DIR, 'models', 'model_metadata.json')) as f:
        metadata = json.load(f)
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'processed', 'features_matrix.csv'))
    df['date'] = pd.to_datetime(df['date'])
    return model, metadata, df

MODEL, METADATA, DF = load_all()

FEATURE_COLS = METADATA['features']

# Pre-compute predictions for all matches
def compute_all_predictions():
    """Add prediction columns to dataframe."""
    X = DF[FEATURE_COLS]
    proba = MODEL.predict_proba(X)
    DF['pred_home_win'] = proba[:, 2]
    DF['pred_draw'] = proba[:, 1]
    DF['pred_away_win'] = proba[:, 0]
    DF['pred_label'] = MODEL.predict(X)

compute_all_predictions()


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictor')
def predictor():
    return render_template('predictor.html')


@app.route('/future')
def future():
    return render_template('future.html')


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/api/model-metrics')
def api_model_metrics():
    """Return model evaluation metrics."""
    return jsonify(METADATA['metrics'])


@app.route('/api/feature-importance')
def api_feature_importance():
    """Return feature importance data."""
    fi = METADATA['feature_importance']
    # Sort by importance
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    return jsonify({
        'labels': [item[0] for item in sorted_fi],
        'values': [item[1] for item in sorted_fi],
    })


@app.route('/api/confusion-matrix')
def api_confusion_matrix():
    """Return confusion matrix for test set."""
    test = DF[DF['date'] >= '2024-01-01']
    y_true = test['result'].values
    y_pred = test['pred_label'].values

    labels = [0, 1, 2]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return jsonify({
        'matrix': cm.tolist(),
        'labels': ['Away Win', 'Draw', 'Home Win'],
    })


@app.route('/api/result-distribution')
def api_result_distribution():
    """Return actual vs predicted result distribution."""
    test = DF[DF['date'] >= '2024-01-01']
    actual = test['result'].value_counts().to_dict()
    predicted = test['pred_label'].value_counts().to_dict()

    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    return jsonify({
        'labels': ['Away Win', 'Draw', 'Home Win'],
        'actual': [actual.get(i, 0) for i in range(3)],
        'predicted': [predicted.get(i, 0) for i in range(3)],
    })


@app.route('/api/probability-calibration')
def api_probability_calibration():
    """Return data for probability calibration chart."""
    test = DF[DF['date'] >= '2024-01-01'].copy()

    # Bin predicted home win probability
    bins = np.arange(0, 1.05, 0.1)
    test['prob_bin'] = pd.cut(test['pred_home_win'], bins=bins)
    cal = test.groupby('prob_bin', observed=True).agg(
        mean_pred=('pred_home_win', 'mean'),
        actual_rate=('result', lambda x: (x == 2).mean()),
        count=('result', 'size'),
    ).dropna()

    return jsonify({
        'predicted': cal['mean_pred'].round(3).tolist(),
        'actual': cal['actual_rate'].round(3).tolist(),
        'counts': cal['count'].tolist(),
    })


@app.route('/api/teams')
def api_teams():
    """Return list of all teams."""
    teams = sorted(set(DF['home_team'].unique()) | set(DF['away_team'].unique()))
    return jsonify(teams)


@app.route('/api/matches')
def api_matches():
    """Return matches filtered by teams and/or tournament."""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    tournament = request.args.get('tournament', '')
    year = request.args.get('year', '')

    filtered = DF.copy()
    if home:
        filtered = filtered[filtered['home_team'] == home]
    if away:
        filtered = filtered[filtered['away_team'] == away]
    if tournament:
        filtered = filtered[filtered['tournament'].str.contains(tournament, case=False, na=False)]
    if year:
        filtered = filtered[filtered['date'].dt.year == int(year)]

    # Sort by date desc, limit 100
    filtered = filtered.sort_values('date', ascending=False).head(100)

    matches = []
    for _, row in filtered.iterrows():
        matches.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']),
            'away_score': int(row['away_score']),
            'tournament': row['tournament'],
            'result': int(row['result']),
            'pred_home_win': round(float(row['pred_home_win']), 3),
            'pred_draw': round(float(row['pred_draw']), 3),
            'pred_away_win': round(float(row['pred_away_win']), 3),
            'pred_label': int(row['pred_label']),
            'correct': int(row['result']) == int(row['pred_label']),
        })

    return jsonify(matches)


@app.route('/api/match-features')
def api_match_features():
    """Return features for a specific match by home, away, date."""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    date = request.args.get('date', '')

    if not (home and away and date):
        return jsonify({'error': 'Missing home, away, or date'}), 400

    match = DF[(DF['home_team'] == home) & (DF['away_team'] == away) &
               (DF['date'].dt.strftime('%Y-%m-%d') == date)]

    if len(match) == 0:
        return jsonify({'error': 'Match not found'}), 404

    row = match.iloc[0]
    return jsonify({
        'features': {col: round(float(row[col]), 4) for col in FEATURE_COLS},
    })


@app.route('/api/tournaments')
def api_tournaments():
    """Return list of tournaments."""
    tournaments = sorted(DF['tournament'].unique().tolist())
    return jsonify(tournaments)


@app.route('/api/years')
def api_years():
    """Return list of years."""
    years = sorted(DF['date'].dt.year.unique().tolist(), reverse=True)
    return jsonify(years)


@app.route('/api/predict-match')
def api_predict_match():
    """Predict a specific match by index or features."""
    idx = request.args.get('idx')
    if idx:
        row = DF.iloc[int(idx)]
        features = {col: float(row[col]) for col in FEATURE_COLS}
        X = pd.DataFrame([features])[FEATURE_COLS]
        proba = MODEL.predict_proba(X)[0]

        return jsonify({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'tournament': row['tournament'],
            'actual_home_score': int(row['home_score']),
            'actual_away_score': int(row['away_score']),
            'actual_result': int(row['result']),
            'pred_home_win': round(float(proba[2]), 4),
            'pred_draw': round(float(proba[1]), 4),
            'pred_away_win': round(float(proba[0]), 4),
            'features': {col: round(float(row[col]), 4) for col in FEATURE_COLS},
        })

    return jsonify({'error': 'Missing idx parameter'}), 400


@app.route('/api/overall-stats')
def api_overall_stats():
    """Return overall model stats."""
    test = DF[DF['date'] >= '2024-01-01']
    total = len(test)
    correct = (test['result'] == test['pred_label']).sum()

    return jsonify({
        'total_matches': len(DF),
        'train_matches': len(DF[DF['date'] < '2024-01-01']),
        'test_matches': total,
        'test_accuracy': round(correct / total, 4),
        'features_count': len(FEATURE_COLS),
        'best_params': METADATA['best_params'],
    })


# ============================================================
# FUTURE MATCH PREDICTION - compute live features for any 2 teams
# ============================================================

# Pre-compute latest stats per team from full dataset
def build_team_stats():
    """Build latest stats for each team from the full dataset."""
    from collections import defaultdict

    # Load Elo ratings
    with open(os.path.join(PROJECT_DIR, 'data', 'processed', 'elo_ratings.json')) as f:
        elo_dict = json.load(f)

    # Build latest FIFA ranking lookup (from the latest row per team)
    rank_lookup = {}
    for _, row in DF.sort_values('date').iterrows():
        rank_lookup[row['home_team']] = {
            'rank': row.get('rank_home', 100),
            'points': row.get('points_home', 500),
        }
        rank_lookup[row['away_team']] = {
            'rank': row.get('rank_away', 100),
            'points': row.get('points_away', 500),
        }

    # Build rolling stats: last N matches per team
    team_goals_scored = defaultdict(list)
    team_goals_conceded = defaultdict(list)
    team_form = defaultdict(list)
    all_goals = []

    for _, row in DF.sort_values('date').iterrows():
        h, a = row['home_team'], row['away_team']
        hs, as_ = row['home_score'], row['away_score']

        team_goals_scored[h].append(hs)
        team_goals_conceded[h].append(as_)
        team_goals_scored[a].append(as_)
        team_goals_conceded[a].append(hs)
        all_goals.append(hs)
        all_goals.append(as_)

        if hs > as_:
            team_form[h].append(3)
            team_form[a].append(0)
        elif hs == as_:
            team_form[h].append(1)
            team_form[a].append(1)
        else:
            team_form[h].append(0)
            team_form[a].append(3)

    avg_goals = np.mean(all_goals[-1000:]) if len(all_goals) > 20 else 1.4

    return elo_dict, rank_lookup, team_goals_scored, team_goals_conceded, team_form, avg_goals


ELO_DICT, RANK_LOOKUP, TEAM_SCORED, TEAM_CONCEDED, TEAM_FORM, AVG_GOALS = build_team_stats()


def compute_future_features(home_team, away_team, is_neutral):
    """Compute the 8 features for a future match between any two teams."""
    # Elo
    home_elo = ELO_DICT.get(home_team, 1500)
    away_elo = ELO_DICT.get(away_team, 1500)
    elo_diff = home_elo - away_elo

    # Rank
    home_rank = RANK_LOOKUP.get(home_team, {}).get('rank', 100)
    away_rank = RANK_LOOKUP.get(away_team, {}).get('rank', 100)
    rank_diff = away_rank - home_rank

    # Attack/Defense strength (last 30 matches)
    h_scored = TEAM_SCORED.get(home_team, [])[-30:]
    h_conceded = TEAM_CONCEDED.get(home_team, [])[-30:]
    a_scored = TEAM_SCORED.get(away_team, [])[-30:]
    a_conceded = TEAM_CONCEDED.get(away_team, [])[-30:]

    home_attack = np.mean(h_scored) / max(AVG_GOALS, 0.5) if len(h_scored) >= 5 else 1.0
    away_defense = np.mean(a_conceded) / max(AVG_GOALS, 0.5) if len(a_conceded) >= 5 else 1.0
    home_defense = np.mean(h_conceded) / max(AVG_GOALS, 0.5) if len(h_conceded) >= 5 else 1.0
    away_attack = np.mean(a_scored) / max(AVG_GOALS, 0.5) if len(a_scored) >= 5 else 1.0

    xg_home = home_attack * away_defense * AVG_GOALS
    xg_away = away_attack * home_defense * AVG_GOALS
    expected_goals_diff = xg_home - xg_away

    # Form (last 5)
    h_form_pts = TEAM_FORM.get(home_team, [])[-5:]
    a_form_pts = TEAM_FORM.get(away_team, [])[-5:]
    home_form = np.mean(h_form_pts) / 3.0 if len(h_form_pts) >= 3 else 0.5
    away_form = np.mean(a_form_pts) / 3.0 if len(a_form_pts) >= 3 else 0.5

    features = {
        'elo_diff': round(elo_diff, 2),
        'rank_diff': round(rank_diff, 2),
        'home_attack_strength': round(home_attack, 4),
        'away_defense_strength': round(away_defense, 4),
        'expected_goals_diff': round(expected_goals_diff, 4),
        'home_form': round(home_form, 4),
        'away_form': round(away_form, 4),
        'is_neutral': int(is_neutral),
    }

    extra = {
        'home_elo': round(home_elo, 1),
        'away_elo': round(away_elo, 1),
        'home_rank': home_rank,
        'away_rank': away_rank,
        'home_attack_strength': round(home_attack, 4),
        'home_defense_strength': round(home_defense, 4),
        'away_attack_strength': round(away_attack, 4),
        'away_defense_strength': round(away_defense, 4),
        'xg_home': round(xg_home, 2),
        'xg_away': round(xg_away, 2),
        'home_form': round(home_form, 4),
        'away_form': round(away_form, 4),
        'home_last5': h_form_pts[-5:] if h_form_pts else [],
        'away_last5': a_form_pts[-5:] if a_form_pts else [],
    }

    return features, extra


@app.route('/api/predict-future')
def api_predict_future():
    """Predict a future match between any two teams."""
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    neutral = request.args.get('neutral', '1')

    if not home or not away:
        return jsonify({'error': 'Missing home or away team'}), 400
    if home == away:
        return jsonify({'error': 'Teams must be different'}), 400

    is_neutral = int(neutral)
    features, extra = compute_future_features(home, away, is_neutral)

    X = pd.DataFrame([features])[FEATURE_COLS]
    proba = MODEL.predict_proba(X)[0]
    pred_code = int(np.argmax(proba))
    class_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    # Head-to-head history
    h2h = DF[
        ((DF['home_team'] == home) & (DF['away_team'] == away)) |
        ((DF['home_team'] == away) & (DF['away_team'] == home))
    ].sort_values('date', ascending=False).head(10)

    h2h_history = []
    for _, row in h2h.iterrows():
        h2h_history.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']),
            'away_score': int(row['away_score']),
            'tournament': row['tournament'],
        })

    return jsonify({
        'home_team': home,
        'away_team': away,
        'is_neutral': is_neutral,
        'pred_home_win': round(float(proba[2]), 4),
        'pred_draw': round(float(proba[1]), 4),
        'pred_away_win': round(float(proba[0]), 4),
        'predicted_label': class_map[pred_code],
        'features': features,
        'extra': extra,
        'h2h_history': h2h_history,
    })


@app.route('/api/team-profile')
def api_team_profile():
    """Return a team's recent stats summary."""
    team = request.args.get('team', '')
    if not team:
        return jsonify({'error': 'Missing team'}), 400

    matches = DF[(DF['home_team'] == team) | (DF['away_team'] == team)].sort_values('date', ascending=False)
    last10 = matches.head(10)

    recent = []
    for _, row in last10.iterrows():
        is_home = row['home_team'] == team
        goals_for = int(row['home_score'] if is_home else row['away_score'])
        goals_against = int(row['away_score'] if is_home else row['home_score'])
        if goals_for > goals_against:
            res = 'W'
        elif goals_for == goals_against:
            res = 'D'
        else:
            res = 'L'

        recent.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'opponent': row['away_team'] if is_home else row['home_team'],
            'goals_for': goals_for,
            'goals_against': goals_against,
            'result': res,
            'tournament': row['tournament'],
            'venue': 'Home' if is_home else 'Away',
        })

    return jsonify({
        'team': team,
        'elo': round(ELO_DICT.get(team, 1500), 1),
        'rank': RANK_LOOKUP.get(team, {}).get('rank', 'N/A'),
        'total_matches': len(matches),
        'recent': recent,
    })


if __name__ == '__main__':
    print("Starting web server at http://localhost:5000")
    app.run(debug=True, port=5000)
