# Hướng dẫn cài đặt & chạy chương trình

## Mục lục
- [1. Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
- [2. Cài đặt môi trường](#2-cài-đặt-môi-trường)
- [3. Cấu trúc project](#3-cấu-trúc-project)
- [4. Chạy từng bước](#4-chạy-từng-bước)
- [5. Chạy Web App](#5-chạy-web-app)
- [6. Troubleshooting](#6-troubleshooting)

---

## 1. Yêu cầu hệ thống

| Yêu cầu | Phiên bản |
|----------|-----------|
| OS | Linux / macOS / Windows |
| Miniconda hoặc Anaconda | Bất kỳ phiên bản nào |
| Python | 3.11+ |
| RAM | >= 4 GB |
| Disk | >= 500 MB (data + model) |
| Internet | Cần cho Bước 1 (download data) |

---

## 2. Cài đặt môi trường

### 2.1 Clone project

```bash
git clone <repo-url>
cd xg-boots-football-analysis
```

### 2.2 Tạo môi trường conda

```bash
conda create -n football-analysis python=3.11 -y
conda activate football-analysis
```

### 2.3 Cài đặt dependencies

```bash
pip install pandas==3.0.1 \
            numpy==2.4.3 \
            scikit-learn==1.8.0 \
            xgboost==3.2.0 \
            scipy==1.17.1 \
            requests==2.32.5 \
            beautifulsoup4==4.14.3 \
            lxml==6.0.2 \
            matplotlib==3.10.8 \
            seaborn==0.13.2 \
            joblib==1.5.3 \
            flask==3.1.3 \
            kagglehub==1.0.0
```

Hoặc dùng file requirements:

```bash
pip install -r requirements.txt
```

### 2.4 Kiểm tra cài đặt

```bash
python -c "
import pandas, numpy, sklearn, xgboost, flask
print('pandas:', pandas.__version__)
print('numpy:', numpy.__version__)
print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('flask:', flask.__version__)
print('All OK!')
"
```

Output mong đợi:
```
pandas: 3.0.1
numpy: 2.4.3
sklearn: 1.8.0
xgboost: 3.2.0
flask: 3.1.3
All OK!
```

---

## 3. Cấu trúc project

```
xg-boots-football-analysis/
│
├── data/
│   ├── raw/                          ← Dữ liệu gốc (Bước 1 download)
│   │   ├── results.csv                   Kết quả 49,000+ trận đấu quốc tế
│   │   ├── goalscorers.csv               Thông tin người ghi bàn
│   │   ├── shootouts.csv                 Kết quả đá penalty
│   │   ├── fifa_ranking-2023-07-20.csv   FIFA Ranking (từ Kaggle)
│   │   ├── fifa_ranking-2024-04-04.csv
│   │   └── fifa_ranking-2024-06-20.csv
│   │
│   └── processed/                    ← Dữ liệu đã xử lý (Bước 2 tạo)
│       ├── features_matrix.csv           20,148 trận × 30 cột (23 features)
│       └── elo_ratings.json              Điểm Elo hiện tại của tất cả đội
│
├── models/                           ← Model đã train (Bước 3 tạo)
│   ├── xgboost_model.joblib              Trained XGBoost model
│   └── model_metadata.json               Params, metrics, feature importance
│
├── src/                              ← Source code chính
│   ├── step1_download_data.py            Bước 1: Download data
│   ├── step2_preprocessing_and_features.py   Bước 2: Feature engineering
│   └── step3_xgboost_model.py            Bước 3: Train XGBoost
│
├── web/                              ← Web application
│   ├── app.py                            Flask backend + API
│   ├── templates/
│   │   ├── index.html                    Dashboard (model metrics)
│   │   ├── predictor.html                Match Predictor (trận trong DB)
│   │   └── future.html                   Future Predictor (trận tương lai)
│   └── static/                           (CSS/JS nếu cần)
│
├── docs/                             ← Tài liệu
│   ├── step2_analysis_data.md            Phân tích Bước 2
│   ├── step3_xgboost_model.md            Phân tích Bước 3
│   └── installation_guide.md             Hướng dẫn cài đặt (file này)
│
├── notebooks/                        ← Jupyter notebooks (dự phòng)
└── data_analysis.docx                ← Tài liệu yêu cầu gốc
```

---

## 4. Chạy từng bước

> **Lưu ý:** Luôn activate conda env trước khi chạy:
> ```bash
> conda activate football-analysis
> ```

### Bước 1: Download data

```bash
python src/step1_download_data.py
```

Script sẽ download từ GitHub:
- `results.csv` — 49,000+ kết quả trận đấu quốc tế (3.5 MB)
- `goalscorers.csv` — thông tin người ghi bàn (3.1 MB)
- `shootouts.csv` — kết quả đá penalty (0.03 MB)

**FIFA Ranking:** Cần tải thủ công từ Kaggle:
1. Truy cập: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
2. Download và giải nén các file `fifa_ranking-*.csv`
3. Copy vào `data/raw/`

Output mong đợi:
```
=== [1/2] International Football Results ===
  -> Saved: results.csv (3.53 MB)
  -> Saved: goalscorers.csv (3.10 MB)
  -> Saved: shootouts.csv (0.03 MB)
=== Download Complete ===
```

### Bước 2: Preprocessing & Feature Engineering

```bash
python src/step2_preprocessing_and_features.py
```

Script sẽ:
1. Load & clean data (lọc từ năm 2000)
2. Tính Elo ratings cho tất cả đội
3. Merge FIFA Ranking (từ file CSV)
4. Tính Poisson attack/defense strength (rolling 30 trận)
5. Tính form features (rolling 5 trận)
6. Tính head-to-head win rate
7. Xuất `features_matrix.csv` + `elo_ratings.json`

Output mong đợi:
```
=== Bước 1b: Preprocessing ===
  Total matches (2000+): 25013
=== Bước 2a: Elo Ratings ===
  Top 20 teams by Elo:
     1. Spain                     2191
     2. Argentina                 2154
     ...
=== Computing Rank Diff (FIFA World Ranking) ===
  FIFA Ranking loaded: 69326 entries, 228 teams
  Matched 25013/25013 matches with FIFA ranking
=== Assembling Final Feature Matrix ===
  Final dataset shape: (20148, 30)
  Missing values: 0
=== Pipeline Complete! Ready for Bước 3: XGBoost ===
```

**Thời gian chạy:** ~1-2 phút

### Bước 3: Train XGBoost Model

```bash
python src/step3_xgboost_model.py
```

Script sẽ:
1. Load features_matrix.csv, chọn 8 features
2. Chia train/test theo thời gian (cutoff: 2024-01-01)
3. GridSearchCV + TimeSeriesSplit (5 folds, 36 tổ hợp params)
4. Train final model với best params
5. Đánh giá: accuracy, F1, confusion matrix, log loss
6. Xuất feature importance
7. Lưu model + metadata

Output mong đợi:
```
=== [3/6] Train XGBoost ===
  Class weights: {0: 1.16, 1: 1.44, 2: 0.69}
  Best params: {max_depth: 3, learning_rate: 0.01, n_estimators: 500, ...}
=== [4/6] Evaluate Model ===
  Test Accuracy:  0.5661
  Test F1 (macro): 0.5222
=== [5/6] Feature Importance ===
  elo_diff        44.0%
  rank_diff       20.2%
  ...
=== Bước 3 Complete! ===
```

**Thời gian chạy:** ~3-5 phút (GridSearch chiếm phần lớn)

### Chạy tất cả (pipeline đầy đủ)

```bash
conda activate football-analysis
python src/step1_download_data.py
python src/step2_preprocessing_and_features.py
python src/step3_xgboost_model.py
```

---

## 5. Chạy Web App

### Khởi động server

```bash
conda activate football-analysis
python web/app.py
```

Output:
```
Starting web server at http://localhost:5000
 * Running on http://127.0.0.1:5000
```

### Mở trình duyệt

| Trang | URL | Mô tả |
|-------|-----|-------|
| Dashboard | http://localhost:5000/ | Metrics, Feature Importance, Confusion Matrix |
| Match Predictor | http://localhost:5000/predictor | Chọn trận trong DB, so sánh dự đoán vs thực tế |
| Future Predictor | http://localhost:5000/future | Dự đoán trận tương lai bất kỳ |

### Tắt server

Nhấn `Ctrl + C` trong terminal.

---

## 6. Troubleshooting

### Lỗi `ModuleNotFoundError: No module named 'xgboost'`

Chưa activate conda env:
```bash
conda activate football-analysis
```

### Lỗi `FileNotFoundError: results.csv`

Chưa chạy Bước 1:
```bash
python src/step1_download_data.py
```

### Lỗi `FileNotFoundError: features_matrix.csv`

Chưa chạy Bước 2:
```bash
python src/step2_preprocessing_and_features.py
```

### Lỗi `FileNotFoundError: xgboost_model.joblib`

Chưa chạy Bước 3:
```bash
python src/step3_xgboost_model.py
```

### FIFA Ranking không có / rank_diff toàn = 0

Cần tải FIFA Ranking từ Kaggle thủ công:
1. https://www.kaggle.com/datasets/cashncarry/fifaworldranking
2. Đặt file `fifa_ranking-*.csv` vào `data/raw/`
3. Chạy lại Bước 2

### Web app hiện lỗi trắng

Kiểm tra terminal có lỗi Python không. Đảm bảo đã chạy đủ Bước 1 → 2 → 3 trước khi chạy web.

### Port 5000 đã bị chiếm

```bash
# Tìm process chiếm port
lsof -i :5000
# Kill process
kill -9 <PID>
# Hoặc dùng port khác
python -c "from web.app import app; app.run(port=8080)"
```

---

*Cập nhật lần cuối: 2026-03-17*
