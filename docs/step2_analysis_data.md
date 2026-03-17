# Bước 2: Analysis Data — Feature Engineering

## Mục lục
- [1. Tổng quan](#1-tổng-quan)
- [2. Nhóm chỉ số Elo Rating](#2-nhóm-chỉ-số-elo-rating)
- [3. Nhóm chỉ số FIFA Ranking](#3-nhóm-chỉ-số-fifa-ranking)
- [4. Nhóm chỉ số Poisson (Attack/Defense Strength)](#4-nhóm-chỉ-số-poisson-attackdefense-strength)
- [5. Nhóm chỉ số Phong độ (Form)](#5-nhóm-chỉ-số-phong-độ-form)
- [6. Nhóm chỉ số Đối đầu (Head-to-Head)](#6-nhóm-chỉ-số-đối-đầu-head-to-head)
- [7. Nhóm chỉ số Bối cảnh (Context)](#7-nhóm-chỉ-số-bối-cảnh-context)
- [8. Phân tích bản ghi mẫu: Chung kết World Cup 2022](#8-phân-tích-bản-ghi-mẫu-chung-kết-world-cup-2022)
- [9. Thống kê tổng hợp](#9-thống-kê-tổng-hợp)

---

## 1. Tổng quan

Bước 2 nhận dữ liệu thô từ Bước 1 (25,013 trận đấu quốc tế từ năm 2000) và tính toán **23 features** chia thành 6 nhóm chính. Sau khi loại bỏ giai đoạn cold-start (2000–2004), output cuối cùng gồm **20,148 trận đấu** từ 2005–2026.

### Pipeline xử lý

```
results.csv (raw)
  → Preprocessing (lọc, tạo label)
    → Elo Rating (tính từ lịch sử trận đấu)
      → FIFA Ranking (merge từ dataset ranking)
        → Poisson Strength (rolling 30 trận)
          → Form Features (rolling 5 trận)
            → Head-to-Head (lịch sử đối đầu)
              → features_matrix.csv (output)
```

### Nguyên tắc chống Data Leakage

Tất cả features đều được tính **TRƯỚC** trận đấu diễn ra. Dữ liệu của trận hiện tại chỉ được cập nhật **SAU** khi đã ghi nhận giá trị feature. Điều này đảm bảo model không "nhìn thấy tương lai".

---

## 2. Nhóm chỉ số Elo Rating

**Mục đích:** Đo lường sức mạnh tổng thể của đội tuyển dựa trên toàn bộ lịch sử thi đấu.

### Thuật toán

Hệ thống Elo xuất phát từ cờ vua, được áp dụng cho bóng đá với các điều chỉnh:

- **Elo khởi điểm:** 1500 cho mọi đội
- **K-factor:** K = 30 (mặc định), nhân 1.5 cho giải chính thức (World Cup, Euro, Copa América...)
- **Home advantage:** +100 điểm cho đội nhà (không áp dụng sân trung lập)
- **Goal difference multiplier:** Thắng cách biệt lớn → thay đổi Elo nhiều hơn
  - Chênh lệch ≤ 1 bàn: ×1.0
  - Chênh lệch = 2 bàn: ×1.5
  - Chênh lệch ≥ 3 bàn: ×(11 + goal_diff) / 8

### Công thức cập nhật

```
Expected_home = 1 / (1 + 10^((Elo_away - Elo_home_adjusted) / 400))

Elo_new = Elo_old + K × GD_mult × (Actual - Expected)
```

Trong đó `Actual` = 1 (thắng), 0.5 (hòa), 0 (thua).

### Features

| Feature | Kiểu | Mô tả |
|---------|------|-------|
| `elo_home` | float | Điểm Elo của đội nhà trước trận |
| `elo_away` | float | Điểm Elo của đội khách trước trận |
| `elo_diff` | float | `elo_home - elo_away`. Dương = đội nhà mạnh hơn |

### Phân phối

- **Mean Elo:** ~1,575 (home nhỉnh hơn away do lợi thế sân nhà qua nhiều trận)
- **Range:** 803 – 2,231
- **Top 5 hiện tại:** Spain (2191), Argentina (2154), France (2074), Colombia (2064), England (2061)

---

## 3. Nhóm chỉ số FIFA Ranking

**Mục đích:** Bổ sung góc nhìn chính thức từ FIFA — khác biệt với Elo vì FIFA dùng công thức riêng và cập nhật theo chu kỳ.

### Nguồn dữ liệu

- 3 file FIFA ranking từ Kaggle: `fifa_ranking-2023-07-20.csv`, `fifa_ranking-2024-04-04.csv`, `fifa_ranking-2024-06-20.csv`
- Tổng cộng **69,326 entries**, **228 đội**, từ 1992 đến 06/2024
- Mapping tên đội được thực hiện để khớp 2 dataset (ví dụ: `USA` → `United States`, `Korea Republic` → `South Korea`)

### Cách merge

Với mỗi trận đấu, tìm bản xếp hạng FIFA **gần nhất trước ngày thi đấu** bằng binary search. Đảm bảo 100% trận đấu được match (fill median cho đội không có trong FIFA ranking).

### Features

| Feature | Kiểu | Mô tả |
|---------|------|-------|
| `rank_home` | int | Thứ hạng FIFA của đội nhà (1 = tốt nhất) |
| `rank_away` | int | Thứ hạng FIFA của đội khách |
| `rank_diff` | int | `rank_away - rank_home`. Dương = đội nhà xếp hạng tốt hơn |
| `points_home` | float | Điểm FIFA của đội nhà |
| `points_away` | float | Điểm FIFA của đội khách |
| `points_diff` | float | `points_home - points_away`. Dương = đội nhà nhiều điểm hơn |

### Tại sao cần cả Elo lẫn FIFA Ranking?

| Tiêu chí | Elo | FIFA Ranking |
|-----------|-----|-------------|
| Nguồn gốc | Tự tính từ kết quả | Chính thức từ FIFA |
| Cập nhật | Sau mỗi trận | Theo chu kỳ (hàng tháng) |
| Trọng số giải đấu | Có (competitive × 1.5) | Có (phức tạp hơn) |
| Ưu điểm | Phản ánh nhanh thay đổi phong độ | Được công nhận rộng rãi, ổn định |

Hai hệ thống **tương quan cao nhưng không hoàn toàn giống nhau**, giúp model có thêm thông tin bổ sung.

---

## 4. Nhóm chỉ số Poisson (Attack/Defense Strength)

**Mục đích:** Đo lường khả năng ghi bàn (tấn công) và khả năng giữ sạch lưới (phòng thủ) của từng đội, dựa trên mô hình Poisson.

### Thuật toán

Dựa trên **rolling window 30 trận gần nhất** của mỗi đội:

```
attack_strength  = avg_goals_scored_by_team  / avg_goals_per_match_globally
defense_strength = avg_goals_conceded_by_team / avg_goals_per_match_globally
```

- `avg_goals_per_match_globally`: trung bình bàn thắng trong 500 trận gần nhất toàn cầu
- Nếu đội chưa đủ 5 trận lịch sử → mặc định = 1.0 (trung bình)

### Expected Goals (Poisson Lambda)

```
expected_goals_home = home_attack × away_defense × avg_goals
expected_goals_away = away_attack × home_defense × avg_goals
```

Đây là **số bàn thắng kỳ vọng** (lambda) để đưa vào phân phối Poisson, tính xác suất cho từng tỉ số cụ thể.

### Features

| Feature | Kiểu | Mô tả |
|---------|------|-------|
| `home_attack_strength` | float | Chỉ số tấn công đội nhà. >1 = mạnh hơn TB |
| `home_defense_strength` | float | Chỉ số phòng thủ đội nhà. <1 = phòng thủ tốt hơn TB |
| `away_attack_strength` | float | Chỉ số tấn công đội khách |
| `away_defense_strength` | float | Chỉ số phòng thủ đội khách |
| `expected_goals_home` | float | Số bàn kỳ vọng đội nhà ghi |
| `expected_goals_away` | float | Số bàn kỳ vọng đội khách ghi |
| `expected_goals_diff` | float | `expected_goals_home - expected_goals_away` |

### Cách đọc

- `attack_strength = 1.65` → Đội ghi bàn nhiều hơn trung bình giải 65%
- `defense_strength = 0.34` → Đội chỉ để lọt lưới bằng 34% so với TB giải (phòng thủ cực tốt)
- `defense_strength = 1.5` → Đội để lọt lưới nhiều hơn TB giải 50% (phòng thủ yếu)

---

## 5. Nhóm chỉ số Phong độ (Form)

**Mục đích:** Đo lường phong độ gần đây (momentum) của đội tuyển trong **5 trận gần nhất**, bất kể giải đấu nào.

### Thuật toán

```
form = mean(points_last_5) / 3.0
```

Trong đó: Thắng = 3 điểm, Hòa = 1 điểm, Thua = 0 điểm. Kết quả được normalize về khoảng [0, 1].

```
goal_diff_5 = mean(goal_difference_last_5)
```

### Features

| Feature | Kiểu | Range | Mô tả |
|---------|------|-------|-------|
| `home_form` | float | 0.0 – 1.0 | Phong độ đội nhà. 1.0 = thắng cả 5 trận |
| `away_form` | float | 0.0 – 1.0 | Phong độ đội khách |
| `goal_diff_home_5` | float | ~(-15, +11) | Hiệu số bàn thắng TB trong 5 trận gần nhất đội nhà |
| `goal_diff_away_5` | float | ~(-15, +7) | Hiệu số bàn thắng TB đội khách |

### Cách đọc

- `form = 0.87` → Khoảng 13 điểm / 15 điểm tối đa (4 thắng + 1 hòa)
- `form = 0.33` → Khoảng 5 điểm / 15 (1 thắng + 2 hòa + 2 thua)
- `goal_diff_5 = +1.6` → Trung bình thắng cách biệt 1.6 bàn trong 5 trận gần nhất

---

## 6. Nhóm chỉ số Đối đầu (Head-to-Head)

**Mục đích:** Đo lường lịch sử đối đầu trực tiếp giữa 2 đội. Một số cặp đấu có "duyên nợ" đặc biệt không thể hiện qua Elo hay ranking.

### Thuật toán

```
h2h_win_rate = số trận đội nhà thắng đội khách / tổng số trận đối đầu
```

- Chỉ tính khi 2 đội đã gặp nhau **ít nhất 3 lần**, nếu không → mặc định 0.5
- Tính từ góc nhìn của đội nhà (home team's perspective)

### Features

| Feature | Kiểu | Range | Mô tả |
|---------|------|-------|-------|
| `h2h_win_rate` | float | 0.0 – 1.0 | Tỉ lệ thắng của đội nhà trong lịch sử đối đầu |

### Cách đọc

- `h2h_win_rate = 0.67` → Đội nhà thắng 2/3 lần đối đầu trực tiếp
- `h2h_win_rate = 0.50` → Cân bằng hoặc chưa đủ dữ liệu

---

## 7. Nhóm chỉ số Bối cảnh (Context)

**Mục đích:** Các yếu tố ngoài sức mạnh đội bóng ảnh hưởng đến kết quả trận đấu.

### Features

| Feature | Kiểu | Giá trị | Mô tả |
|---------|------|---------|-------|
| `is_neutral` | int | 0 hoặc 1 | 1 = sân trung lập (World Cup, Confederations Cup...) |
| `is_competitive` | int | 0 hoặc 1 | 1 = giải chính thức (không phải giao hữu) |

### Tầm quan trọng

- **Sân trung lập:** Loại bỏ lợi thế sân nhà. World Cup 2026 hầu hết là sân trung lập (trừ USA, Mexico, Canada)
- **Giải chính thức:** Các đội thường thi đấu nghiêm túc hơn giao hữu → kết quả đáng tin cậy hơn

### Phân phối trong dataset

- 29.3% trận đấu trên sân trung lập
- 47.3% trận đấu thuộc giải chính thức

---

## 8. Phân tích bản ghi mẫu: Chung kết World Cup 2022

### Argentina vs France — 18/12/2022 (3-3, Argentina thắng penalty)

Bảng dưới đây mô tả **tất cả 30 trường dữ liệu** trong 1 bản ghi của `features_matrix.csv`, chia rõ:
- **Nguồn gốc:** Lấy trực tiếp từ file nào, hoặc được tính bằng công thức gì
- **XGBoost Input:** Trường nào được đưa vào model, trường nào chỉ là metadata

---

### 8.1 Nhóm Metadata (7 trường) — KHÔNG đưa vào XGBoost

Các trường này dùng để **nhận diện trận đấu** và **tạo label**, không phải feature cho model.

| # | Trường | Giá trị mẫu | Nguồn gốc | Vai trò |
|---|--------|-------------|------------|---------|
| 1 | `date` | `2022-12-18` | Lấy trực tiếp từ cột `date` trong `data/raw/results.csv` | Dùng để sắp xếp thời gian, chia train/test |
| 2 | `home_team` | `Argentina` | Lấy trực tiếp từ cột `home_team` trong `results.csv` | Nhận diện đội nhà |
| 3 | `away_team` | `France` | Lấy trực tiếp từ cột `away_team` trong `results.csv` | Nhận diện đội khách |
| 4 | `home_score` | `3` | Lấy trực tiếp từ cột `home_score` trong `results.csv` | Dùng để tính `result`, không đưa vào model (vì đây là kết quả sau trận) |
| 5 | `away_score` | `3` | Lấy trực tiếp từ cột `away_score` trong `results.csv` | Tương tự `home_score` |
| 6 | `tournament` | `FIFA World Cup` | Lấy trực tiếp từ cột `tournament` trong `results.csv` | Dùng để phân loại, không đưa trực tiếp vào model |
| 7 | `result` | `1` (Draw) | **Tính từ:** `home_score` vs `away_score` → 2=Home Win, 1=Draw, 0=Away Win | **TARGET LABEL** — biến mục tiêu mà XGBoost cần dự đoán |

---

### 8.2 Nhóm Elo (3 trường) — INPUT cho XGBoost

| # | Trường | Giá trị mẫu | Nguồn gốc | Công thức / Cách tính |
|---|--------|-------------|------------|----------------------|
| 8 | `elo_home` | `2173.24` | **Tự tính** từ toàn bộ lịch sử `results.csv` (từ năm 2000) | Bắt đầu = 1500. Sau mỗi trận: `Elo_new = Elo_old + K × GD_mult × (Actual - Expected)`. K=30 (×1.5 nếu giải chính thức). Giá trị này là Elo của Argentina **trước** trận chung kết, tích lũy qua hàng trăm trận trước đó |
| 9 | `elo_away` | `2061.26` | **Tự tính** tương tự `elo_home` | Elo của France tích lũy đến trước trận này |
| 10 | `elo_diff` | `+111.97` | **Tính từ:** `elo_home - elo_away` | Dương → đội nhà mạnh hơn. +112 ≈ Argentina thắng 65% trên sân trung lập |

> **Lưu ý:** Elo được tính tuần tự trận-by-trận. Giá trị Elo của 1 trận phụ thuộc vào KẾT QUẢ tất cả trận trước đó, nhưng KHÔNG dùng kết quả trận hiện tại (chống data leakage).

---

### 8.3 Nhóm FIFA Ranking (6 trường) — INPUT cho XGBoost

| # | Trường | Giá trị mẫu | Nguồn gốc | Công thức / Cách tính |
|---|--------|-------------|------------|----------------------|
| 11 | `rank_home` | `3` | Lấy từ `data/raw/fifa_ranking-*.csv` | Tra cứu cột `rank` theo tên đội (`country_full`) + ngày gần nhất ≤ ngày trận đấu (`rank_date` ≤ `2022-12-18`). Mapping tên: `Korea Republic` → `South Korea`, `USA` → `United States`... |
| 12 | `rank_away` | `4` | Tương tự `rank_home` | Tra cứu rank của France tại thời điểm gần nhất trước trận |
| 13 | `rank_diff` | `+1` | **Tính từ:** `rank_away - rank_home` | Dương → đội nhà xếp hạng tốt hơn (số nhỏ = hạng cao). France hạng 4, Argentina hạng 3 → 4-3 = +1 |
| 14 | `points_home` | `1773.88` | Lấy từ cột `total_points` trong `fifa_ranking-*.csv` | Điểm FIFA chính thức của Argentina tại kỳ ranking gần nhất |
| 15 | `points_away` | `1759.78` | Tương tự `points_home` | Điểm FIFA của France |
| 16 | `points_diff` | `+14.10` | **Tính từ:** `points_home - points_away` | Dương → đội nhà nhiều điểm FIFA hơn. Chênh lệch rất nhỏ (+14/1773 ≈ 0.8%) |

> **So sánh Elo vs FIFA:** Elo cho thấy khoảng cách lớn (+112), FIFA Ranking gần ngang (+1 hạng, +14 điểm). Cả 2 hệ thống cùng đưa vào model để XGBoost tự học cái nào quan trọng hơn.

---

### 8.4 Nhóm Poisson Strength (7 trường) — INPUT cho XGBoost

| # | Trường | Giá trị mẫu | Nguồn gốc | Công thức / Cách tính |
|---|--------|-------------|------------|----------------------|
| 17 | `home_attack_strength` | `1.6458` | **Tự tính** từ `results.csv` | `mean(goals_scored_by_Argentina_in_last_30_matches) / mean(all_goals_last_500_matches)`. Argentina ghi TB 2.3 bàn/trận ÷ TB giải 1.4 = **1.65** → tấn công mạnh hơn TB 65% |
| 18 | `home_defense_strength` | `0.3396` | **Tự tính** từ `results.csv` | `mean(goals_conceded_by_Argentina_in_last_30_matches) / mean(all_goals_last_500_matches)`. Argentina để lọt lưới TB 0.47 bàn/trận ÷ TB giải 1.4 = **0.34** → chỉ thủng lưới bằng 34% so với TB |
| 19 | `away_attack_strength` | `1.5935` | **Tự tính** tương tự | France ghi bàn nhiều hơn TB giải 59% |
| 20 | `away_defense_strength` | `0.6531` | **Tự tính** tương tự | France để thủng lưới bằng 65% so với TB (phòng thủ tốt, nhưng kém Argentina) |
| 21 | `expected_goals_home` | `1.3715` | **Tính từ:** `home_attack × away_defense × avg_goals` | `1.6458 × 0.6531 × 1.4 = 1.37` → mô hình kỳ vọng Argentina ghi ~1.4 bàn |
| 22 | `expected_goals_away` | `0.6905` | **Tính từ:** `away_attack × home_defense × avg_goals` | `1.5935 × 0.3396 × 1.4 = 0.69` → mô hình kỳ vọng France ghi ~0.7 bàn |
| 23 | `expected_goals_diff` | `+0.6809` | **Tính từ:** `expected_goals_home - expected_goals_away` | `1.37 - 0.69 = +0.68` → Argentina kỳ vọng ghi nhiều hơn ~0.7 bàn |

> **Rolling window = 30 trận:** Mỗi chỉ số dựa trên 30 trận gần nhất của đội đó (bất kể đá nhà hay khách, bất kể giải nào). Nếu đội chưa đủ 5 trận → mặc định = 1.0.

---

### 8.5 Nhóm Form (4 trường) — INPUT cho XGBoost

| # | Trường | Giá trị mẫu | Nguồn gốc | Công thức / Cách tính |
|---|--------|-------------|------------|----------------------|
| 24 | `home_form` | `0.8667` | **Tự tính** từ `results.csv` | Lấy 5 trận gần nhất của Argentina trước 18/12/2022. Tính điểm: Thắng=3, Hòa=1, Thua=0. `form = mean(points) / 3`. VD: 4 thắng + 1 hòa = (3+3+3+3+1)/5 = 2.6 → 2.6/3 = **0.867** |
| 25 | `away_form` | `0.8000` | **Tự tính** tương tự | 5 trận gần nhất của France: 4 thắng + 1 thua = (3+3+3+3+0)/5 = 2.4 → 2.4/3 = **0.80** |
| 26 | `goal_diff_home_5` | `+1.6` | **Tự tính** từ `results.csv` | `mean(goal_diff_last_5_matches_of_Argentina)`. VD: nếu 5 trận gần nhất thắng 2-1, 2-0, 3-0, 1-0, 2-1 → hiệu số = +1,+2,+3,+1,+1 → TB = **+1.6** |
| 27 | `goal_diff_away_5` | `+1.0` | **Tự tính** tương tự | Hiệu số bàn thắng TB 5 trận gần nhất của France |

> **Rolling window = 5 trận:** Phong độ ngắn hạn, phản ánh momentum hiện tại. Khác Poisson (30 trận) phản ánh năng lực dài hạn.

---

### 8.6 Nhóm H2H (1 trường) — INPUT cho XGBoost

| # | Trường | Giá trị mẫu | Nguồn gốc | Công thức / Cách tính |
|---|--------|-------------|------------|----------------------|
| 28 | `h2h_win_rate` | `0.6667` | **Tự tính** từ `results.csv` | Đếm tất cả trận Argentina vs France **trước** 18/12/2022 trong dataset (từ năm 2000). `h2h = số_trận_Argentina_thắng / tổng_số_trận`. VD: gặp nhau 3 lần, Argentina thắng 2 → 2/3 = **0.667**. Nếu < 3 trận đối đầu → mặc định 0.5 |

> **Góc nhìn:** Luôn tính từ perspective của `home_team`. Nếu trận khác France là home, giá trị sẽ khác.

---

### 8.7 Nhóm Context (2 trường) — INPUT cho XGBoost

| # | Trường | Giá trị mẫu | Nguồn gốc | Công thức / Cách tính |
|---|--------|-------------|------------|----------------------|
| 29 | `is_neutral` | `1` | Lấy trực tiếp từ cột `neutral` trong `results.csv`, convert True/False → 1/0 | 1 = sân trung lập (Qatar). World Cup đa số sân trung lập trừ đội chủ nhà |
| 30 | `is_competitive` | `1` | **Tự tính** từ cột `tournament` trong `results.csv` | Kiểm tra tên giải có nằm trong danh sách giải chính thức không: FIFA World Cup, UEFA Euro, Copa América, AFC Asian Cup, African Cup of Nations, UEFA Nations League... → 1 = có, 0 = không (giao hữu) |

---

### 8.8 Sơ đồ tổng hợp: Nguồn gốc → Feature → XGBoost

```
┌─────────────────────────┐     ┌──────────────────────────┐
│   data/raw/results.csv  │     │ data/raw/fifa_ranking-   │
│                         │     │ *.csv (từ Kaggle)        │
│ • date                  │     │                          │
│ • home_team, away_team  │     │ • rank, total_points     │
│ • home_score,away_score │     │ • country_full           │
│ • tournament            │     │ • rank_date              │
│ • neutral               │     │                          │
└────────┬────────────────┘     └────────────┬─────────────┘
         │                                   │
         │ Preprocessing                     │ Merge by team+date
         ▼                                   ▼
┌────────────────────────────────────────────────────────────┐
│              step2_preprocessing_and_features.py           │
│                                                            │
│  ┌─ LẤY TRỰC TIẾP ──────────────────────────────────────┐ │
│  │ date, home_team, away_team, home_score, away_score,   │ │
│  │ tournament, is_neutral                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ TÍNH TỪ home_score/away_score ───────────────────────┐ │
│  │ result (target label: 0/1/2)                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ TÍNH TUẦN TỰ QUA TỪNG TRẬN (Elo algorithm) ─────────┐ │
│  │ elo_home, elo_away, elo_diff                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ MERGE TỪ FIFA RANKING CSV ───────────────────────────┐ │
│  │ rank_home, rank_away, rank_diff                       │ │
│  │ points_home, points_away, points_diff                 │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ TÍNH ROLLING 30 TRẬN (Poisson logic) ────────────────┐ │
│  │ home/away_attack_strength, home/away_defense_strength │ │
│  │ expected_goals_home/away/diff                         │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ TÍNH ROLLING 5 TRẬN ─────────────────────────────────┐ │
│  │ home_form, away_form, goal_diff_home_5/away_5         │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ TÍNH TỪ LỊCH SỬ ĐỐI ĐẦU ───────────────────────────┐ │
│  │ h2h_win_rate                                          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ TÍNH TỪ tournament name ─────────────────────────────┐ │
│  │ is_competitive                                        │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│            data/processed/features_matrix.csv              │
│                                                            │
│  METADATA (7 cột) ─── Không đưa vào XGBoost               │
│  │ date, home_team, away_team, home_score, away_score,    │
│  │ tournament, result (= TARGET LABEL)                    │
│  │                                                        │
│  ALL FEATURES (23 cột) ─── Có sẵn trong features_matrix   │
│  │ elo_home, elo_away, elo_diff                     [Elo] │
│  │ rank_home, rank_away, rank_diff             [FIFA Rank] │
│  │ points_home, points_away, points_diff       [FIFA Pts]  │
│  │ home/away_attack_strength                   [Poisson]  │
│  │ home/away_defense_strength                  [Poisson]  │
│  │ expected_goals_home/away/diff               [Poisson]  │
│  │ home_form, away_form                          [Form]   │
│  │ goal_diff_home_5, goal_diff_away_5            [Form]   │
│  │ h2h_win_rate                                   [H2H]   │
│  │ is_neutral, is_competitive                 [Context]   │
│  │                                                        │
│  XGBOOST INPUT (8 cột) ─── Đầu vào chính thức Bước 3     │
│  │ ① elo_diff                           [Sức mạnh]       │
│  │ ② rank_diff                          [Sức mạnh]       │
│  │ ③ home_attack_strength               [Poisson]        │
│  │ ④ away_defense_strength              [Poisson]        │
│  │ ⑤ expected_goals_diff                [Poisson]        │
│  │ ⑥ home_form                          [Form]           │
│  │ ⑦ away_form                          [Form]           │
│  │ ⑧ is_neutral                         [Bối cảnh]       │
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│              Bước 3: XGBoost Model                         │
│                                                            │
│  X = 8 features      →  XGBClassifier  →   Predictions    │
│  y = result (0/1/2)     (multi:softprob)   [P_win,        │
│                                             P_draw,        │
│                                             P_loss]        │
└────────────────────────────────────────────────────────────┘
```

---

### 8.9 Tổng kết bản ghi mẫu

**Trận chung kết Argentina vs France:** Mọi chỉ số đều nghiêng về Argentina — Elo cao hơn (+112), FIFA ranking tốt hơn (hạng 3 vs 4), tấn công mạnh hơn (1.65 vs 1.59), phòng thủ chắc hơn (0.34 vs 0.65), phong độ tốt hơn (0.87 vs 0.80), lịch sử đối đầu áp đảo (67%). Tuy nhiên kết quả thực tế là **hòa 3-3** — Mbappé lập hat-trick giúp France cân bằng.

Điều này minh họa:
1. **Bóng đá có tính bất định cao** — model chỉ dự đoán xác suất, không phải kết quả chắc chắn
2. **XGBoost nên output xác suất** (ví dụ: P(Win)=55%, P(Draw)=25%, P(Loss)=20%) thay vì chỉ 1 nhãn
3. **Monte Carlo simulation** (Bước 4) cần chạy hàng ngàn lần để phản ánh sự biến động này

---

## 9. Thống kê tổng hợp

### Dataset cuối cùng

| Thông số | Giá trị |
|----------|---------|
| Tổng số trận | 20,148 |
| Khoảng thời gian | 01/2005 – 01/2026 |
| Số đội tuyển | 312 |
| Số features | 23 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Phân phối kết quả (Target Variable)

| Kết quả | Số trận | Tỉ lệ |
|---------|---------|--------|
| Home Win (result=2) | 9,656 | 47.9% |
| Draw (result=1) | 4,677 | 23.2% |
| Away Win (result=0) | 5,815 | 28.9% |

> **Lưu ý:** Class imbalance nhẹ — Home Win chiếm gần 50%. XGBoost (Bước 3) cần cân nhắc class weights hoặc stratified sampling.

### Bảng tổng hợp Features

| # | Feature | Nhóm | Min | Mean | Max | Mô tả |
|---|---------|------|-----|------|-----|-------|
| 1 | `elo_home` | Elo | 803 | 1,582 | 2,225 | Điểm Elo đội nhà |
| 2 | `elo_away` | Elo | 805 | 1,564 | 2,232 | Điểm Elo đội khách |
| 3 | `elo_diff` | Elo | -1,221 | +18 | +1,186 | Chênh lệch Elo |
| 4 | `rank_home` | FIFA | 1 | 83 | 211 | Hạng FIFA đội nhà |
| 5 | `rank_away` | FIFA | 1 | 86 | 211 | Hạng FIFA đội khách |
| 6 | `rank_diff` | FIFA | -210 | +3 | +209 | Chênh lệch hạng |
| 7 | `points_home` | FIFA | 0 | 796 | 2,164 | Điểm FIFA đội nhà |
| 8 | `points_away` | FIFA | 0 | 780 | 2,164 | Điểm FIFA đội khách |
| 9 | `points_diff` | FIFA | -1,757 | +16 | +1,596 | Chênh lệch điểm FIFA |
| 10 | `home_attack_strength` | Poisson | 0.00 | 1.02 | 4.60 | Chỉ số tấn công đội nhà |
| 11 | `home_defense_strength` | Poisson | 0.16 | 0.99 | 7.21 | Chỉ số phòng thủ đội nhà |
| 12 | `away_attack_strength` | Poisson | 0.00 | 0.99 | 4.22 | Chỉ số tấn công đội khách |
| 13 | `away_defense_strength` | Poisson | 0.16 | 1.03 | 11.02 | Chỉ số phòng thủ đội khách |
| 14 | `expected_goals_home` | Poisson | 0.00 | 1.40 | 17.63 | Bàn thắng kỳ vọng đội nhà |
| 15 | `expected_goals_away` | Poisson | 0.00 | 1.31 | 16.29 | Bàn thắng kỳ vọng đội khách |
| 16 | `expected_goals_diff` | Poisson | -16.14 | +0.09 | +17.21 | Chênh lệch bàn thắng kỳ vọng |
| 17 | `home_form` | Form | 0.00 | 0.47 | 1.00 | Phong độ đội nhà (5 trận) |
| 18 | `away_form` | Form | 0.00 | 0.46 | 1.00 | Phong độ đội khách (5 trận) |
| 19 | `goal_diff_home_5` | Form | -15.2 | +0.04 | +11.0 | Hiệu số bàn thắng TB (5 trận) đội nhà |
| 20 | `goal_diff_away_5` | Form | -14.6 | -0.04 | +6.2 | Hiệu số bàn thắng TB (5 trận) đội khách |
| 21 | `h2h_win_rate` | H2H | 0.00 | 0.44 | 1.00 | Tỉ lệ thắng đối đầu trực tiếp |
| 22 | `is_neutral` | Context | 0 | 0.29 | 1 | Sân trung lập |
| 23 | `is_competitive` | Context | 0 | 0.47 | 1 | Giải chính thức |

---

## Output Files

| File | Đường dẫn | Mô tả |
|------|-----------|-------|
| Feature Matrix | `data/processed/features_matrix.csv` | Ma trận 20,148 × 30 (7 metadata + 23 features) |
| Elo Ratings | `data/processed/elo_ratings.json` | Điểm Elo hiện tại của tất cả đội (dùng cho Bước 3/4) |

---

*Tài liệu được tạo tự động từ pipeline `src/step2_preprocessing_and_features.py`*
