# ⚽ XG-Boots Football Analysis

Hệ thống dự đoán kết quả bóng đá quốc tế sử dụng **XGBoost** — phân tích 20,000+ trận đấu, dự đoán xác suất Thắng/Hòa/Thua cho World Cup 2026.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)
![Flask](https://img.shields.io/badge/Flask-3.1.3-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📊 Kết quả

| Metric | Giá trị |
|--------|---------|
| Test Accuracy | **56.61%** |
| Baseline Accuracy | 47.13% |
| Improvement | **+9.48%** |
| F1 Score (macro) | 0.5222 |
| Log Loss | 0.891 |

### Feature Importance

```
elo_diff              ████████████████████████████████████████████  44.0%
rank_diff             ████████████████████                          20.3%
expected_goals_diff   ██████████                                    10.8%
is_neutral            █████████                                      9.9%
away_defense_strength ████                                           5.1%
home_attack_strength  ███                                            3.9%
home_form             ██                                             3.2%
away_form             ██                                             2.8%
```

---

## 🏗️ Pipeline

```
Bước 1: Download Data ──→ Bước 2: Feature Engineering ──→ Bước 3: XGBoost Model
     results.csv              Elo Rating, FIFA Ranking         Train 500 trees
     goalscorers.csv          Poisson Strength                 GridSearchCV (36 combos)
     fifa_ranking-*.csv       Form, Head-to-Head               Class Weights
                              → features_matrix.csv            → xgboost_model.joblib
                                (20,148 trận × 30 cột)
```

---

## 🚀 Quickstart

### 1. Cài đặt

```bash
# Clone
git clone <repo-url>
cd xg-boots-football-analysis

# Tạo môi trường
conda create -n football-analysis python=3.11 -y
conda activate football-analysis

# Cài dependencies
pip install -r requirements.txt
```

### 2. Chạy Pipeline

```bash
python src/step1_download_data.py                    # Download data
python src/step2_preprocessing_and_features.py       # Feature engineering
python src/step3_xgboost_model.py                    # Train model
```

> ⚠️ **FIFA Ranking** cần tải thủ công từ [Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) → đặt vào `data/raw/`

### 3. Chạy Web App

```bash
python web/app.py
# Mở http://localhost:5000
```

---

## 🌐 Web Application

### Dashboard (`/`)
Tổng quan model: accuracy, feature importance (click xem giải thích), confusion matrix, result distribution.

### Match Predictor (`/predictor`)
Chọn trận đấu trong DB → xem dự đoán vs kết quả thực tế, xác suất 3 kết quả, bảng features.

### Future Predictor (`/future`)
Dự đoán trận tương lai: chọn 2 đội bất kỳ → xác suất, so sánh chỉ số, lịch sử đối đầu, phong độ gần đây.

---

## 📁 Cấu trúc Project

```
xg-boots-football-analysis/
├── data/
│   ├── raw/                          ← Dữ liệu gốc
│   └── processed/                    ← features_matrix.csv, elo_ratings.json
├── models/                           ← xgboost_model.joblib, model_metadata.json
├── src/
│   ├── step1_download_data.py        ← Download data
│   ├── step2_preprocessing_and_features.py  ← Feature engineering
│   └── step3_xgboost_model.py        ← Train XGBoost
├── web/
│   ├── app.py                        ← Flask backend + API
│   └── templates/                    ← Dashboard, Predictor, Future
├── docs/                             ← Tài liệu chi tiết
└── requirements.txt
```

---

## 📖 Tài liệu

| File | Nội dung |
|------|----------|
| [docs/project_overview.md](docs/project_overview.md) | Tổng quan project |
| [docs/step2_analysis_data.md](docs/step2_analysis_data.md) | Phân tích Feature Engineering |
| [docs/step3_xgboost_model.md](docs/step3_xgboost_model.md) | Phân tích XGBoost Model |
| [docs/installation_guide.md](docs/installation_guide.md) | Hướng dẫn cài đặt chi tiết |

---

## 🔧 Tech Stack

- **ML:** XGBoost, scikit-learn, pandas, numpy
- **Web:** Flask, Chart.js
- **Data:** Elo Rating System, Poisson Model, FIFA Ranking
- **Viz:** matplotlib, seaborn

---

## 👥 Đóng góp

Project này là một phần của bài phân tích dữ liệu bóng đá, bao gồm 5 bước:

| Bước | Nội dung | Trạng thái |
|------|----------|-----------|
| Bước 1 | Crawling & Preprocessing | ✅ Hoàn thành |
| Bước 2 | Analysis Data (Feature Engineering) | ✅ Hoàn thành |
| Bước 3 | XGBoost Model | ✅ Hoàn thành |
| Bước 4 | Monte Carlo Simulation | 🔲 Chưa triển khai |
| Bước 5 | Dashboard tổng hợp | 🔲 Chưa triển khai |
