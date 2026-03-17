"""
Bước 3: XGBoost Model
- Input: 8 features (elo_diff, rank_diff, home_attack_strength,
         away_defense_strength, expected_goals_diff, home_form, away_form, is_neutral)
- Output: [P(Win), P(Draw), P(Loss)] cho mỗi trận đấu
- Target: result (2=Home Win, 1=Draw, 0=Away Win)
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, log_loss
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 8 features theo đề bài
FEATURE_COLS = [
    'elo_diff',                 # Chênh lệch Elo
    'rank_diff',                # Chênh lệch hạng FIFA
    'home_attack_strength',     # Sức tấn công đội nhà (Poisson)
    'away_defense_strength',    # Sức phòng thủ đội khách (Poisson)
    'expected_goals_diff',      # Chênh lệch bàn thắng kỳ vọng
    'home_form',                # Phong độ đội nhà (5 trận)
    'away_form',                # Phong độ đội khách (5 trận)
    'is_neutral',               # Sân trung lập
]

TARGET_COL = 'result'  # 2=Home Win, 1=Draw, 0=Away Win


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    """Load features_matrix.csv and select 8 features."""
    print("=== [1/6] Load Data ===")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'features_matrix.csv'))
    df['date'] = pd.to_datetime(df['date'])

    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Target: {TARGET_COL} -> {df[TARGET_COL].value_counts().to_dict()}")

    return df


# ============================================================
# 2. TRAIN/TEST SPLIT (time-based)
# ============================================================

def split_data(df, test_start='2024-01-01'):
    """Split by time: train < test_start, test >= test_start."""
    print(f"\n=== [2/6] Train/Test Split (cutoff: {test_start}) ===")

    train = df[df['date'] < test_start].copy()
    test = df[df['date'] >= test_start].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    print(f"  Train: {len(train)} matches ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"  Test:  {len(test)} matches ({test['date'].min().date()} -> {test['date'].max().date()})")
    print(f"  Train result distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test  result distribution: {y_test.value_counts().to_dict()}")

    return X_train, y_train, X_test, y_test, train, test


# ============================================================
# 3. TRAIN XGBOOST
# ============================================================

def train_model(X_train, y_train):
    """Train XGBoost with hyperparameter tuning via TimeSeriesSplit CV."""
    print("\n=== [3/6] Train XGBoost ===")

    # Compute sample weights to handle class imbalance (Draw is underrepresented)
    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    class_weight = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
    sample_weights = y_train.map(class_weight).values
    weights_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(class_weight.items()))
    print(f"  Class weights: {{{weights_str}}}")

    # --- Phase 1: GridSearch for best hyperparameters ---
    print("  Phase 1: Hyperparameter tuning (GridSearchCV + TimeSeriesSplit)...")

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3],
    }

    base_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    print(f"  Best params: {best_params}")
    print(f"  Best CV log_loss: {best_score:.4f}")

    # --- Phase 2: Train final model with best params ---
    print("\n  Phase 2: Training final model with best params...")

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
        **best_params,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("  Model trained successfully.")
    return model, best_params


# ============================================================
# 4. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model: accuracy, F1, confusion matrix, log loss."""
    print("\n=== [4/6] Evaluate Model ===")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)
    y_proba_test = model.predict_proba(X_test)

    # --- Train metrics ---
    print("  --- Train Set ---")
    print(f"  Accuracy:  {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"  F1 (macro): {f1_score(y_train, y_pred_train, average='macro'):.4f}")
    print(f"  Log Loss:  {log_loss(y_train, y_proba_train):.4f}")

    # --- Test metrics ---
    print("\n  --- Test Set ---")
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    test_logloss = log_loss(y_test, y_proba_test)

    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  F1 (macro): {test_f1:.4f}")
    print(f"  Log Loss:  {test_logloss:.4f}")

    # --- Classification Report ---
    label_names = ['Away Win (0)', 'Draw (1)', 'Home Win (2)']
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=label_names, digits=3))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"  Confusion Matrix (Test):")
    print(f"  {'':>15s} Predicted:")
    print(f"  {'':>15s} {'Away':>8s} {'Draw':>8s} {'Home':>8s}")
    for i, label in enumerate(label_names):
        print(f"  Actual {label}: {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    # --- Baseline comparison ---
    print(f"\n  --- Baseline Comparison ---")
    # Baseline 1: always predict Home Win (most frequent class)
    baseline_acc = (y_test == 2).mean()
    print(f"  Always predict Home Win: {baseline_acc:.4f}")
    # Baseline 2: predict based on elo_diff sign
    print(f"  XGBoost improvement over baseline: +{(test_acc - baseline_acc)*100:.1f}%")

    metrics = {
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'test_log_loss': test_logloss,
        'baseline_accuracy': baseline_acc,
    }

    return metrics


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(model):
    """Extract and display feature importance."""
    print("\n=== [5/6] Feature Importance ===")

    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importance,
        'importance_pct': (importance / importance.sum() * 100),
    }).sort_values('importance', ascending=False)

    print("  Feature Importance (XGBoost gain):")
    print(f"  {'Feature':>30s} | {'Importance':>10s} | {'%':>6s} | Bar")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*6}-+{'-'*30}")

    max_imp = feat_imp['importance_pct'].max()
    for _, row in feat_imp.iterrows():
        bar_len = int(row['importance_pct'] / max_imp * 25)
        bar = '█' * bar_len
        print(f"  {row['feature']:>30s} | {row['importance']:.6f} | {row['importance_pct']:5.1f}% | {bar}")

    return feat_imp


# ============================================================
# 6. SAVE MODEL & PREDICTION FUNCTION
# ============================================================

def save_model(model, best_params, metrics, feat_imp):
    """Save trained model and metadata."""
    print("\n=== [6/6] Save Model ===")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    # Save metadata
    metadata = {
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'classes': {0: 'Away Win', 1: 'Draw', 2: 'Home Win'},
        'best_params': best_params,
        'metrics': {k: round(v, 4) for k, v in metrics.items()},
        'feature_importance': feat_imp.set_index('feature')['importance_pct'].round(2).to_dict(),
    }
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    return model_path


def predict_match(model, features_dict):
    """
    Predict a single match result.

    Args:
        model: trained XGBClassifier
        features_dict: dict with 8 feature values
            e.g. {'elo_diff': 112, 'rank_diff': 1, ...}

    Returns:
        dict: {
            'probabilities': {'Home Win': 0.65, 'Draw': 0.20, 'Away Win': 0.15},
            'predicted_label': 'Home Win',
            'predicted_code': 2
        }
    """
    X = pd.DataFrame([features_dict])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]

    class_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    pred_code = int(np.argmax(proba))

    return {
        'probabilities': {
            'Home Win': round(float(proba[2]), 4),
            'Draw': round(float(proba[1]), 4),
            'Away Win': round(float(proba[0]), 4),
        },
        'predicted_label': class_map[pred_code],
        'predicted_code': pred_code,
    }


def demo_predictions(model, df):
    """Demo: predict some well-known matches from test set."""
    print("\n=== Demo Predictions ===")

    # Pick some interesting matches from recent data
    recent = df[df['date'] >= '2024-01-01'].copy()

    interesting_matches = []
    # Try to find big matches
    big_teams = ['Argentina', 'France', 'Brazil', 'Germany', 'Spain',
                 'England', 'Portugal', 'Netherlands', 'Italy']
    for _, row in recent.iterrows():
        if row['home_team'] in big_teams and row['away_team'] in big_teams:
            interesting_matches.append(row)
        if len(interesting_matches) >= 5:
            break

    # If not enough big matches, take last 5
    if len(interesting_matches) < 3:
        interesting_matches = [row for _, row in recent.tail(5).iterrows()]

    for match in interesting_matches:
        features = {col: match[col] for col in FEATURE_COLS}
        result = predict_match(model, features)
        actual_map = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
        actual = actual_map[match['result']]
        actual_score = f"{int(match['home_score'])}-{int(match['away_score'])}"

        print(f"\n  {match['home_team']} vs {match['away_team']} ({match['date'].date()}, {match['tournament']})")
        print(f"    Actual: {actual} ({actual_score})")
        print(f"    Predicted: {result['predicted_label']}")
        probs = result['probabilities']
        print(f"    P(Home Win)={probs['Home Win']:.1%}  P(Draw)={probs['Draw']:.1%}  P(Away Win)={probs['Away Win']:.1%}")
        correct = '✓' if result['predicted_label'] == actual else '✗'
        print(f"    {correct}")


# ============================================================
# MAIN
# ============================================================

def main():
    # 1. Load
    df = load_data()

    # 2. Split
    X_train, y_train, X_test, y_test, train_df, test_df = split_data(df)

    # 3. Train
    model, best_params = train_model(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 5. Feature Importance
    feat_imp = compute_feature_importance(model)

    # 6. Save
    save_model(model, best_params, metrics, feat_imp)

    # Demo predictions
    demo_predictions(model, df)

    print("\n=== Bước 3 Complete! Model ready for Bước 4: Monte Carlo ===")
    return model


if __name__ == '__main__':
    main()
