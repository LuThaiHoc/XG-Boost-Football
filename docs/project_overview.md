# Tổng quan Project: XG-Boots Football Analysis

## Mục lục
- [1. Giới thiệu](#1-giới-thiệu)
- [2. Mục tiêu](#2-mục-tiêu)
- [3. Dữ liệu](#3-dữ-liệu)
- [4. Phương pháp](#4-phương-pháp)
- [5. Kết quả](#5-kết-quả)
- [6. Web Application](#6-web-application)
- [7. Cấu trúc Project](#7-cấu-trúc-project)
- [8. Kết luận & Hướng phát triển](#8-kết-luận--hướng-phát-triển)

---

## 1. Giới thiệu

**XG-Boots Football Analysis** là hệ thống dự đoán kết quả bóng đá quốc tế sử dụng thuật toán **XGBoost** (eXtreme Gradient Boosting). Hệ thống phân tích dữ liệu lịch sử hơn **49,000 trận đấu quốc tế**, tính toán các chỉ số sức mạnh đội tuyển, và đưa ra dự đoán xác suất cho 3 kết quả: **Thắng nhà (Home Win)**, **Hòa (Draw)**, **Thắng khách (Away Win)**.

Project hướng tới ứng dụng cho **World Cup 2026**, cung cấp nền tảng dự đoán dựa trên dữ liệu khoa học thay vì cảm tính.

---

## 2. Mục tiêu

### 2.1 Mục tiêu chính

- Xây dựng mô hình **phân loại đa lớp (multi-class classification)** dự đoán kết quả trận đấu bóng đá quốc tế
- Đạt **độ chính xác vượt trội so với baseline** (dự đoán ngẫu nhiên theo phân phối kết quả)
- Cung cấp **xác suất** cho từng kết quả, không chỉ là dự đoán đơn lẻ

### 2.2 Mục tiêu phụ

- Xác định **features quan trọng nhất** ảnh hưởng đến kết quả trận đấu
- Xây dựng **web application** trực quan hóa kết quả và cho phép dự đoán trận tương lai
- Tạo nền tảng dữ liệu cho các bước tiếp theo (Monte Carlo Simulation, Dashboard tổng hợp)

---

## 3. Dữ liệu

### 3.1 Nguồn dữ liệu

| Dataset | Nguồn | Mô tả | Kích thước |
|---------|-------|-------|------------|
| International Football Results | GitHub (martj42) | Kết quả 49,000+ trận quốc tế từ 1872 | 3.53 MB |
| Goalscorers | GitHub (martj42) | Thông tin người ghi bàn | 3.10 MB |
| Shootouts | GitHub (martj42) | Kết quả đá penalty | 0.03 MB |
| FIFA World Ranking | Kaggle (cashncarry) | Bảng xếp hạng FIFA chính thức | 69,326 entries, 228 đội |

### 3.2 Phạm vi dữ liệu

- **Thời gian:** 2000–2026 (25,013 trận sau khi lọc)
- **Sau feature engineering:** 2005–2026 (20,148 trận — loại giai đoạn cold-start 2000–2004)
- **Số đội tuyển:** 200+ đội tuyển quốc gia
- **Loại giải:** World Cup, Continental (Euro, Copa América, AFCON...), Qualifiers, Friendlies

### 3.3 Phân phối kết quả

| Kết quả | Số trận | Tỷ lệ |
|---------|---------|--------|
| Home Win | ~8,800 | ~43.7% |
| Draw | ~4,600 | ~22.8% |
| Away Win | ~6,700 | ~33.5% |

> Draw là lớp thiểu số — model cần xử lý class imbalance để dự đoán tốt.

---

## 4. Phương pháp

### 4.1 Pipeline tổng thể

```
Bước 1: Download Data        Bước 2: Feature Engineering        Bước 3: XGBoost Model
┌─────────────────┐     ┌──────────────────────────┐     ┌────────────────────────┐
│ results.csv     │     │ Elo Rating               │     │ Train/Test Split       │
│ goalscorers.csv │ ──→ │ FIFA Ranking             │ ──→ │ GridSearchCV           │
│ shootouts.csv   │     │ Poisson Strength         │     │ Train Final Model      │
│ fifa_ranking-*  │     │ Form Features            │     │ Evaluate & Export      │
└─────────────────┘     │ Head-to-Head             │     └────────────────────────┘
                        └──────────────────────────┘
                         Output: features_matrix.csv      Output: xgboost_model.joblib
                                 (20,148 × 30)                    model_metadata.json
```

### 4.2 Feature Engineering (Bước 2)

Từ dữ liệu thô, tính toán **23 features** thuộc 6 nhóm. Model XGBoost sử dụng **8 features** chính:

| # | Feature | Nhóm | Mô tả | Importance |
|---|---------|------|-------|------------|
| 1 | `elo_diff` | Elo Rating | Chênh lệch điểm Elo (home - away) | **44.0%** |
| 2 | `rank_diff` | FIFA Ranking | Chênh lệch thứ hạng FIFA (away - home) | **20.3%** |
| 3 | `expected_goals_diff` | Poisson | Chênh lệch bàn thắng kỳ vọng | **10.8%** |
| 4 | `is_neutral` | Context | Sân trung lập (1) hay sân nhà (0) | **9.9%** |
| 5 | `away_defense_strength` | Poisson | Sức mạnh phòng thủ đội khách | **5.1%** |
| 6 | `home_attack_strength` | Poisson | Sức mạnh tấn công đội nhà | **3.9%** |
| 7 | `home_form` | Form | Phong độ gần đây đội nhà (5 trận) | **3.2%** |
| 8 | `away_form` | Form | Phong độ gần đây đội khách (5 trận) | **2.8%** |

> **Insight:** Riêng `elo_diff` + `rank_diff` đã chiếm **64.3%** tầm quan trọng — sức mạnh tổng thể của đội tuyển là yếu tố quyết định nhất.

#### Nguyên tắc chống Data Leakage

Tất cả features được tính **TRƯỚC** trận đấu diễn ra. Dữ liệu của trận hiện tại chỉ được cập nhật **SAU** khi đã ghi nhận feature. Điều này đảm bảo model không "nhìn thấy tương lai" khi training.

### 4.3 XGBoost Model (Bước 3)

#### Thuật toán

XGBoost sử dụng **500 cây quyết định nhỏ (depth=3)**, mỗi cây **sửa lỗi** của các cây trước đó. Kết quả cuối cùng là tổng hợp xác suất từ tất cả 500 cây.

```
Cây #1: nhìn elo_diff  → dự đoán sơ bộ
Cây #2: sửa lỗi cây #1 → nhìn is_neutral
Cây #3: sửa lỗi cây #2 → nhìn away_defense
...
Cây #500: tinh chỉnh cuối cùng
→ Output: P(Home Win), P(Draw), P(Away Win)
```

#### Hyperparameters (tìm bằng GridSearchCV)

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `n_estimators` | 500 | Số lượng cây quyết định |
| `max_depth` | 3 | Độ sâu tối đa mỗi cây (giữ đơn giản) |
| `learning_rate` | 0.01 | Tốc độ học (nhỏ → học chậm nhưng chính xác) |
| `subsample` | 0.8 | Mỗi cây chỉ dùng 80% dữ liệu (chống overfit) |
| `colsample_bytree` | 0.8 | Mỗi cây chỉ dùng 80% features |
| `min_child_weight` | 1 | Số mẫu tối thiểu trong 1 node lá |

#### Xử lý Class Imbalance

Draw chiếm tỷ lệ thấp nhất (~22.8%), dẫn đến model ban đầu **không bao giờ dự đoán Draw** (recall = 0%). Giải pháp: áp dụng **class weights** theo công thức:

```
weight(class) = total_samples / (n_classes × count_per_class)

→ Away Win: 1.16  (phạt nhẹ — lớp trung bình)
→ Draw:     1.44  (phạt nặng — lớp thiểu số, buộc model chú ý hơn)
→ Home Win: 0.69  (giảm — lớp đa số)
```

Kết quả: Draw recall tăng từ **0% → 27.2%**.

#### Training Process

- **Train/Test split theo thời gian:** Train < 2024-01-01 (17,901 trận) | Test ≥ 2024-01-01 (2,247 trận)
- **Cross-validation:** TimeSeriesSplit (5 folds) — giữ đúng thứ tự thời gian
- **GridSearchCV:** 36 tổ hợp hyperparameters × 5 folds = 180 lần training
- **Metric tối ưu:** F1 macro (cân bằng giữa các lớp)

---

## 5. Kết quả

### 5.1 Metrics tổng hợp

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **Test Accuracy** | **56.61%** | Tỷ lệ dự đoán đúng trên tập test |
| **Baseline Accuracy** | 47.13% | Nếu luôn đoán lớp phổ biến nhất (Home Win) |
| **Improvement** | **+9.48%** | Model tốt hơn baseline 9.48 điểm phần trăm |
| **F1 macro** | **0.5222** | Trung bình F1 của 3 lớp (cân bằng) |
| **Log Loss** | **0.891** | Chất lượng xác suất dự đoán (thấp = tốt) |

### 5.2 Chi tiết theo lớp

| Lớp | Precision | Recall | F1 | Nhận xét |
|-----|-----------|--------|----|----------|
| **Home Win** | 0.60 | 0.74 | 0.66 | Dự đoán tốt nhất — lớp đa số |
| **Draw** | 0.35 | 0.27 | 0.31 | Khó nhất — bóng đá vốn khó đoán hòa |
| **Away Win** | 0.63 | 0.57 | 0.60 | Khá tốt |

### 5.3 Confusion Matrix

```
                 Predicted
                 Away  Draw  Home
Actual  Away   [ 430    98   227 ]    57% recall
        Draw   [ 196   140   181 ]    27% recall
        Home   [ 60    162   753 ]    77% recall
```

### 5.4 Đánh giá

- **56.61% accuracy** là kết quả hợp lý cho bài toán dự đoán bóng đá 3 lớp. Nghiên cứu học thuật thường đạt 50–60% cho bài toán tương tự.
- **Draw vẫn khó đoán** (F1 = 0.31) — đây là thách thức chung trong dự đoán bóng đá vì hòa phụ thuộc nhiều vào yếu tố ngẫu nhiên trong trận.
- **Home Win dễ đoán nhất** (recall 77%) — lợi thế sân nhà là pattern rõ ràng nhất mà model nắm bắt được.
- Model **vượt baseline 9.48%** — chứng tỏ các features thực sự mang thông tin hữu ích.

### 5.5 Ví dụ dự đoán: Chung kết World Cup 2022

**Argentina vs France** (sân trung lập, Qatar):

| Feature | Giá trị | Giải thích |
|---------|---------|-----------|
| elo_diff | +112 | Argentina Elo cao hơn |
| rank_diff | +1 | Argentina hạng 3, France hạng 4 |
| home_attack_strength | 1.65 | Argentina tấn công mạnh |
| away_defense_strength | 0.65 | France phòng thủ tốt |
| expected_goals_diff | +0.68 | Argentina kỳ vọng ghi nhiều hơn |
| home_form | 0.87 | Argentina phong độ cao |
| away_form | 0.80 | France phong độ tốt |
| is_neutral | 1 | Sân trung lập |

**Kết quả dự đoán:**
```
P(Home Win) = 32.5%
P(Draw)     = 36.0%  ← cao nhất → Prediction: Draw
P(Away Win) = 31.5%

Thực tế: Draw (3-3 sau 120 phút) ✓ ĐÚNG
```

---

## 6. Web Application

Hệ thống cung cấp **3 trang web** được xây dựng bằng Flask + Chart.js:

### 6.1 Dashboard (`/`)

Trang tổng quan hiển thị hiệu suất model:

- **4 thẻ thống kê:** Total Matches, Accuracy, F1 Score, Log Loss
- **Feature Importance:** Biểu đồ thanh ngang — click vào từng feature để xem giải thích chi tiết (công thức, ví dụ, cách tính)
- **Confusion Matrix:** Heatmap so sánh dự đoán vs thực tế
- **Result Distribution:** So sánh phân phối thực tế vs dự đoán
- **Probability Calibration:** Đánh giá chất lượng xác suất
- **Model Parameters:** Bảng hyperparameters

### 6.2 Match Predictor (`/predictor`)

Tra cứu và đánh giá dự đoán cho các trận đấu **đã diễn ra** trong cơ sở dữ liệu:

- **Bộ lọc:** Đội nhà, Đội khách, Giải đấu, Năm
- **Danh sách trận:** Badge Correct/Wrong cho từng trận
- **Panel chi tiết:** Xác suất 3 kết quả (thanh + donut chart), bảng features, so sánh dự đoán vs thực tế

### 6.3 Future Predictor (`/future`)

Dự đoán kết quả cho **trận đấu tương lai bất kỳ:**

- **Chọn 2 đội** + tùy chọn sân trung lập + nút Swap
- **Biểu đồ xác suất:** Gauge bar 3 màu + donut chart
- **So sánh đội:** Bảng chỉ số (Elo, Rank, Form, Attack, Defense) với highlight xanh/xám
- **Lịch sử đối đầu:** 10 trận gần nhất
- **Phong độ gần đây:** 10 trận gần nhất mỗi đội (W/D/L dots)

### 6.4 API Endpoints

Web app cung cấp **15 REST API endpoints** để phục vụ dữ liệu:

| Endpoint | Mô tả |
|----------|-------|
| `/api/model-metrics` | Accuracy, F1, Log Loss |
| `/api/feature-importance` | Tầm quan trọng 8 features |
| `/api/confusion-matrix` | Ma trận nhầm lẫn |
| `/api/result-distribution` | Phân phối kết quả (actual vs predicted) |
| `/api/probability-calibration` | Dữ liệu calibration |
| `/api/teams` | Danh sách đội tuyển |
| `/api/matches` | Lọc trận đấu theo team/tournament/year |
| `/api/match-features` | Features chi tiết 1 trận |
| `/api/tournaments` | Danh sách giải đấu |
| `/api/years` | Danh sách năm |
| `/api/predict-match` | Dự đoán 1 trận trong DB |
| `/api/overall-stats` | Thống kê tổng accuracy |
| `/api/predict-future` | Dự đoán trận tương lai |
| `/api/team-profile` | Thông tin chi tiết 1 đội |

---

## 7. Cấu trúc Project

```
xg-boots-football-analysis/
│
├── data/
│   ├── raw/                              ← Dữ liệu gốc (Bước 1)
│   │   ├── results.csv                       49,000+ kết quả trận đấu
│   │   ├── goalscorers.csv                   Thông tin người ghi bàn
│   │   ├── shootouts.csv                     Kết quả đá penalty
│   │   └── fifa_ranking-*.csv                FIFA Ranking (3 files)
│   │
│   └── processed/                        ← Dữ liệu đã xử lý (Bước 2)
│       ├── features_matrix.csv               20,148 trận × 30 cột
│       └── elo_ratings.json                  Điểm Elo tất cả đội
│
├── models/                               ← Model đã train (Bước 3)
│   ├── xgboost_model.joblib                  XGBoost trained model
│   └── model_metadata.json                   Params, metrics, importance
│
├── src/                                  ← Source code pipeline
│   ├── step1_download_data.py                Download data từ GitHub
│   ├── step2_preprocessing_and_features.py   Feature engineering (23 features)
│   └── step3_xgboost_model.py                Train XGBoost (8 features)
│
├── web/                                  ← Web application
│   ├── app.py                                Flask backend + 15 API endpoints
│   └── templates/
│       ├── index.html                        Dashboard
│       ├── predictor.html                    Match Predictor
│       └── future.html                       Future Predictor
│
├── docs/                                 ← Tài liệu
│   ├── project_overview.md                   Tổng quan project (file này)
│   ├── step2_analysis_data.md                Phân tích Feature Engineering
│   ├── step3_xgboost_model.md                Phân tích XGBoost Model
│   └── installation_guide.md                 Hướng dẫn cài đặt
│
└── requirements.txt                      ← Dependencies
```

---

## 8. Kết luận & Hướng phát triển

### 8.1 Kết luận

- Hệ thống đạt **56.61% accuracy**, vượt baseline **9.48%** — chứng tỏ các features được tính toán mang thông tin dự đoán thực sự.
- **Elo Rating** và **FIFA Ranking** là 2 yếu tố quan trọng nhất (chiếm 64.3% importance), cho thấy sức mạnh tổng thể của đội tuyển là predictor chính.
- **Draw** vẫn là thách thức lớn nhất (F1 = 0.31) — đây là bản chất của bóng đá, không phải lỗi model.
- Web application cung cấp giao diện trực quan để khám phá kết quả và dự đoán trận tương lai.

### 8.2 Hướng phát triển

| Bước tiếp theo | Mô tả |
|----------------|-------|
| **Bước 4: Monte Carlo Simulation** | Mô phỏng 10,000+ kịch bản World Cup 2026 dựa trên xác suất từ XGBoost |
| **Bước 5: Dashboard tổng hợp** | Tích hợp kết quả Monte Carlo, hiển thị xác suất vô địch từng đội |
| **Cải thiện model** | Thêm features mới (cầu thủ chấn thương, thời tiết, referee), thử các thuật toán khác (LightGBM, Neural Network) |
| **Cập nhật dữ liệu** | Tự động cập nhật kết quả trận đấu và FIFA ranking mới nhất |

---

*Cập nhật lần cuối: 2026-03-17*
