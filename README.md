# 🏡 Hanoi Real Estate Automated Valuation Model (AVM)

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Models](https://img.shields.io/badge/Models-LightGBM%20%7C%20MLP%20%7C%20HistGBR%20%7C%20DecisionTree%20%7C%20Ridge-green.svg)](https://scikit-learn.org/)
[![Zero Data Leakage](https://img.shields.io/badge/Data%20Leakage-Guaranteed%20Zero-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/pytest-13%20passed%20(100%25)-success.svg)](https://docs.pytest.org/)
[![Uncertainty](https://img.shields.io/badge/Quantile-80%25%20Mondrian%20Conformal-orange.svg)]()

Hệ thống **Định giá Bất động sản Tự động (Automated Valuation Model - AVM)** cấp độ sản xuất dành riêng cho hai phân khúc nhà ở lớn nhất tại **Hà Nội**: **Nhà** (nhà riêng, nhà mặt phố, phân lô, tòa CHDV) và **Căn hộ chung cư**.

Hệ thống cung cấp:
1. **Giá trị giao dịch hợp lý (Point Valuation)**: Tính bằng **VNĐ**, **Tỷ VNĐ** và đơn giá **Triệu VNĐ/m²**.
2. **Khoảng tin cậy định giá 80% (`[P10, P90]`)**: Hiệu chuẩn bằng **Mondrian Split Conformalized Quantile Regression (CQR)**, đảm bảo độ phủ thực tế và triệt tiêu hoàn toàn hiện tượng cắt chéo phân vị.
3. **Giải thích trực quan cục bộ (TreeSHAP Explainer)**: Bóc tách Top 5 yếu tố cốt lõi (dự án, tuyến đường, phân tầng ngõ/phố, kết cấu, tiện ích) thúc đẩy định giá từng căn nhà.
4. **Hệ thống So sánh Đa Thuật toán Toàn diện**: Bao gồm từ mô hình cơ sở Rule-based, Hồi quy tuyến tính Ridge, Cây quyết định đơn lẻ (Decision Tree), Mạng nơ-ron nhân tạo đa tầng (MLP Regressor), đến các mô hình GBDT tiên tiến (HistGBR, LightGBM Chuyên gia và Tổ hợp Ensemble).

---

## 📑 Mục Lục
- [1. Điểm Nhấn Kiến Trúc & Kỹ Thuật](#1-điểm-nhấn-kiến-trúc--kỹ-thuật)
- [2. Kiến Trúc Luồng Dữ Liệu (Medallion Pipeline)](#2-kiến-trúc-luồng-dữ-liệu-medallion-pipeline)
- [3. Hệ Thống Thuật Toán & So Sánh Benchmark (8 Mô Hình)](#3-hệ-thống-thuật-toán--so-sánh-benchmark-8-mô-hình)
- [4. Định Lượng Bất Định (Mondrian Conformal Prediction)](#4-định-lượng-bất-định-mondrian-conformal-prediction)
- [5. Cấu Trúc Thư Mục Dự Án](#5-cấu-trúc-thư-mục-dự-án)
- [6. Hướng Dẫn Cài Đặt & Chạy Toàn Bộ Dự Án](#6-hướng-dẫn-cài-đặt--chạy-toàn-bộ-dự-án)
- [7. Hướng Dẫn Định Giá Thực Tế (Inference CLI & Rich Output)](#7-hướng-dẫn-định-giá-thực-tế-inference-cli--rich-output)
- [8. Kiểm Thử Tự Động (Automated Testing)](#8-kiểm-thử-tự-động-automated-testing)

---

## 1. Điểm Nhấn Kiến Trúc & Kỹ Thuật

* **Tuân thủ Nghiêm ngặt Zero Data Leakage**:
  * **Purged Chronological Walk-Forward Split**: Dữ liệu chia tăng dần theo dòng thời gian (`Train 70%` $\to$ `Val 15%` $\to$ `Test 15%`). Bất kỳ tin đăng nào trong tập Val/Test trùng nhóm tài sản (`property_group_id`) với tập trước đều bị loại bỏ triệt để (**Cross-split Group Purging**).
  * **Stateless Price-Redacted NLP**: Tự động nhận diện và xóa sạch mọi ký tự tiền tệ (`tỷ`, `triệu`, `tr`, `đ`, ...) trước khi trích xuất số tầng, số phòng ngủ, mặt tiền, lộ giới ngõ từ văn bản để triệt tiêu hoàn toàn rò rỉ nhãn mục tiêu.
  * **Hierarchical Out-Of-Fold (OOF) Target Encoding**: Mã hóa biến vị trí 5 cấp (Thành phố $\to$ Loại hình $\to$ Quận $\to$ Phường $\to$ Đường/Dự án) kèm làm mịn Empirical Bayes. Sử dụng 5-Fold OOF trên tập Train để không bản ghi huấn luyện nào nhìn thấy giá trị mục tiêu của chính nó.
* **Hệ Thống Mô Hình Hồi Quy Phân Tầng Đa Dạng**:
  * Đánh giá và so sánh đa dạng họ thuật toán: Cây quyết định đơn lẻ (`Decision Tree`), Mạng nơ-ron sâu (`MLP Regressor`), Hồi quy tuyến tính co cụm (`Ridge`), GBDT dạng phân nhóm (`HistGBR`), và các chuyên gia GBDT tối ưu (`LightGBM Segmented/Unified`).
  * Tinh chỉnh hệ số tỷ lệ nhân vi phân $\alpha$ (Multiplicative Scale Factor) trên tập Validation cho từng mô hình để tối ưu hóa trực tiếp hàm mục tiêu cân bằng sai số tỷ lệ thực tế $0.5 \cdot \text{MAPE} + 0.5 \cdot \text{MdAPE}$.
* **Định Lượng Độ Bất Định Đáng Tin Cậy (Mondrian Conformal Quantile Regression)**:
  * Huấn luyện 2 mô hình LightGBM phân vị tại $\alpha = 0.10$ và $\alpha = 0.90$.
  * Áp dụng kỹ thuật hiệu chuẩn Mondrian Split Conformal trên tập Validation theo từng loại hình để đảm bảo khoảng biến thiên `[P10, P90]` phản ánh đúng mức độ biến động thị trường.
* **Giải Thích Mô Hình Minh Bạch (Explainable AI - TreeSHAP & Rich CLI)**:
  * Tích hợp thuật toán **TreeSHAP** (`pred_contrib=True`) giải thích chính xác đóng góp của từng đặc tính cho từng lượt định giá độc lập.
  * Giao diện Terminal định dạng bảng màu Rich trực quan, chuyên nghiệp.

---

## 2. Kiến Trúc Luồng Dữ Liệu (Medallion Pipeline)

```
[Raw Layer] data/raw/shard_0000.parquet (350,075 dòng toàn quốc - BẤT BIẾN)
   │
   └──► filter_hanoi.py ──► data/raw/hanoi_housing_raw.parquet (93,940 dòng)
           │
[Silver Layer]
           ├──► check_missing.py & handle_missing.py
           │       └──► data/silver/hanoi_missing_handled.parquet (90,938 dòng)
           │
           ├──► check_duplicates.py & remove_duplicates.py
           │       └──► data/silver/hanoi_deduplicated.parquet (90,073 dòng)
           │
           └──► clean_outliers.py (src/data_cleaning.py)
                   │  - Chuẩn hóa hướng (13 -> 9), tên đường
                   │  - Loại ngoại lai sai đơn vị, tin thuê lọt lưới (-444 dòng)
                   │  - Làm sạch ô vật lý bất hợp lý về NaN
                   │  - Khử broker repost gần trùng (-11,976 dòng) & cấp property_group_id
                   ▼
                data/silver/hanoi_cleaned.parquet (77,653 dòng sạch, 76,493 nhóm BĐS)
   │
[Gold Layer]
   └──► split_dataset.py (src/splitting.py)
           │  - Chronological Split (01/06/2025 -> 30/07/2025)
           │  - Cross-Split Purging (-168 dòng Val, -150 dòng Test)
           ▼
        data/gold/train.parquet (54,357 dòng)
        data/gold/val.parquet   (11,480 dòng)
        data/gold/test.parquet  (11,498 dòng - GIỮ NGUYÊN ĐÁNH GIÁ CUỐI)
```

---

## 3. Hệ Thống Thuật Toán & So Sánh Benchmark (8 Mô Hình)

Dự án triển khai và đối sánh **8 mô hình học máy** khác nhau trên cùng một pipeline đặc trưng chống rò rỉ dữ liệu (110 features). 

### Chi tiết các thuật toán được cài đặt (`src/modeling.py`, `src/course_models.py`):
1. **`1_Naive_District_Median`**: Mô hình cơ sở phi tham số dựa trên trung vị đơn giá theo từng cặp `(Loại hình, Quận)`.
2. **`2_Ridge_LogLinear`**: Hồi quy tuyến tính có hiệu chuẩn L2 (`alpha=25.0`) kết hợp chuẩn hóa `StandardScaler` trên không gian log-linear.
3. **`3_Sklearn_HistGBR`**: Thuật toán Histogram-based Gradient Boosting với khả năng xử lý `NaN` tự nhiên và hàm mất mát L1 (Least Absolute Deviation).
4. **`4_LightGBM_Unified`**: Mô hình GBDT toàn cục tối ưu hàm mất mát **Huber Loss** trên thang logarit $\log(\text{price} / \text{area})$ kèm Early Stopping trên tập Validation.
5. **`5_LightGBM_Segmented`**: Hệ thống 2 mô hình GBDT chuyên gia riêng biệt (1 chuyên gia cho `Nhà` và 1 chuyên gia cho `Căn hộ chung cư`).
6. **`6_Champion_LightGBM_Ensemble`**: Mô hình tổ hợp kết hợp hình học trong không gian log-price giữa chuyên gia Segmented ($w=0.70$) và Unified ($w=0.30$).
7. **`7_Decision_Tree_L3`**: Cây quyết định đơn lẻ (`DecisionTreeRegressor`), tìm kiếm siêu tham số tối ưu trên Validation (`max_depth=12, min_samples_leaf=20`, đạt 958 lá).
8. **`8_MLP_Regressor_L6`**: Mạng nơ-ron nhân tạo đa tầng (`MLPRegressor` kiến trúc `[64, 32]`, hàm kích hoạt ReLU, tối ưu hóa Adam, `alpha=0.01`, `max_iter=180`) kết hợp `TransformedTargetRegressor` để chuẩn hóa phân phối mục tiêu đầu ra.

---

### Bảng Kết Quả Đánh Giá Thực Nghiệm Trên Tập Test Độc Lập (`11,498` BĐS)

Toàn bộ mô hình được đánh giá trên tập **Test Holdout** (nửa cuối tháng 7/2025, chưa từng thấy trong quá trình huấn luyện và chọn siêu tham số):

| STT | Mô Hình Thuật Toán | MAPE (%) $\downarrow$ | MdAPE (%) $\downarrow$ | PPE15 (%) $\uparrow$ | PPE20 (%) $\uparrow$ | MAE (Tỷ) $\downarrow$ | RMSE (Tỷ) $\downarrow$ | $R^2_{\log}$ $\uparrow$ | $R^2_{\text{price}}$ $\uparrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `1_Naive_District_Median` | 25.49% | 19.08% | 41.33% | 52.10% | 4.400 | 14.101 | 0.796 | 0.612 |
| 2 | `2_Ridge_LogLinear` | 18.11% | 12.90% | 56.10% | 68.01% | 3.103 | 10.588 | 0.894 | 0.781 |
| 3 | `7_Decision_Tree_L3` *(Mới)* | 18.23% | 12.21% | 57.86% | 69.21% | 3.166 | 9.968 | 0.887 | 0.806 |
| 4 | `8_MLP_Regressor_L6` *(Mới)* | 17.36% | 11.96% | 58.97% | 70.26% | 3.074 | 9.647 | 0.895 | 0.818 |
| 5 | `3_Sklearn_HistGBR` | 15.78% | 10.80% | 63.32% | 74.49% | 2.775 | 9.332 | 0.913 | 0.830 |
| 6 | `4_LightGBM_Unified` | 15.21% | 10.47% | 64.34% | 75.04% | 2.660 | 8.860 | 0.920 | 0.847 |
| 7 | `5_LightGBM_Segmented` | 14.96% | 10.22% | 65.24% | 75.71% | 2.656 | 8.923 | 0.921 | 0.845 |
| 🏆 | **`6_Champion_LightGBM_Ensemble`** | **14.95%** | **10.21%** | **65.22%** | **75.68%** | **2.649** | **8.891** | **0.921** | **0.846** |

> **Nhận xét chính**:
> * **Mạng nơ-ron MLP Regressor (`L6`)** đạt độ chính xác rất khả quan (`MAPE = 17.36%`, `PPE20 = 70.26%`), vượt qua mô hình Tuyến tính Ridge và Cây quyết định đơn lẻ.
> * **Mô hình Vô địch (Champion LightGBM Ensemble)** đạt đỉnh hiệu năng với **`PPE20 = 75.68%`** (>3/4 bất động sản có sai số định giá trong khoảng $\pm 20\%$) và **`MdAPE = 10.21%`** (riêng Chung cư đạt **`MdAPE = 6.65%`** và **`PPE20 = 90.57%`**).

---

## 4. Định Lượng Bất Định (Mondrian Conformal Prediction)

* **Mục tiêu độ phủ lý thuyết**: 80% (`target_coverage = 0.80`).
* **Độ phủ thực nghiệm trên tập Test (`PICP_80`)**: **78.67%** (riêng phân khúc Nhà đạt **79.91%**).
* **Kiểm định tính đơn điệu**: **100%** không vi phạm hiện tượng cắt chéo phân vị (`P10 <= Fair_Price <= P90`).
* **Độ sắc nét thích nghi**: Khoảng giá của Căn hộ chung cư có độ rộng trung vị chỉ **`24.60%`** (khoảng $\pm 12.3\%$), trong khi Nhà đất tự động mở rộng lên **`53.93%`** để phản ánh đúng mức độ biến động đặc thù của nhà ngõ, thửa đất và pháp lý.

---

## 5. Cấu Trúc Thư Mục Dự Án

```
PROPERTY-VALUATION/
├── data/                               # Dữ liệu theo kiến trúc Medallion
│   ├── raw/                            # shard_0000.parquet, hanoi_housing_raw.parquet
│   ├── silver/                         # hanoi_missing_handled, hanoi_deduplicated, hanoi_cleaned
│   └── gold/                           # train.parquet, val.parquet, test.parquet
├── models/                             # Trọng số mô hình đã huấn luyện (.joblib)
│   ├── champion_point_bundle.joblib    # Bundle Pipeline đặc trưng + Mô hình Ensemble
│   └── avm_conformal_system.joblib     # Toàn bộ hệ thống Point + Conformal Quantile
├── reports/                            # Báo cáo benchmark, feature importance, slice analysis
├── src/                                # Package mã nguồn cốt lõi
│   ├── data_cleaning.py                # Xử lý ngoại lai, khử repost, cấp group_id
│   ├── features.py                     # NLP không trạng thái, Hierarchical OOF Target Encoder
│   ├── modeling.py                     # Huấn luyện 8 mô hình hồi quy, Bootstrap CI
│   ├── course_models.py                # Định nghĩa cấu hình & Bundle cho DecisionTree & MLP
│   ├── intervals_and_analysis.py       # Mondrian Conformal CQR, Slice Error Analysis
│   └── splitting.py                    # Purged chronological walk-forward split
├── tests/                              # Bộ kiểm thử tự động pytest (13 test cases)
├── analyze_and_calibrate.py            # Script hiệu chuẩn Conformal Quantile & Slice Report
├── clean_outliers.py                   # Script thực thi làm sạch Giai đoạn 3
├── HUONG_DAN_DU_AN_TUNG_BUOC.md        # Tài liệu thuyết minh chi tiết dự án
├── predict_valuation.py                # Entrypoint CLI Định giá Thực tế + TreeSHAP + Rich UI
├── split_dataset.py                    # Script phân chia dữ liệu Chronological Purged
├── train_models.py                     # Script huấn luyện & so sánh benchmark mô hình
├── pytest.ini                          # Cấu hình kiểm thử pytest
└── requirements.txt                    # Danh sách thư viện phụ thuộc
```

---

## 6. Hướng Dẫn Cài Đặt & Chạy Toàn Bộ Dự Án

### 1. Kích hoạt môi trường ảo
```powershell
# Kích hoạt venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện nếu cần
pip install -r requirements.txt
```

### 2. Chạy toàn bộ Pipeline dữ liệu & Huấn luyện (End-to-End)
Chạy lần lượt các bước từ dữ liệu thô:

```powershell
# Bước 1: Lọc dữ liệu Hà Nội từ Raw Shard
python filter_hanoi.py

# Bước 2: Xử lý khuyết thiếu & Khử trùng lặp sơ bộ
python handle_missing.py
python remove_duplicates.py

# Bước 3: Làm sạch chuyên sâu, chuẩn hóa thực thể & khử tin môi giới đăng lại
python clean_outliers.py

# Bước 4: Chia tập Purged Chronological Split (Train/Val/Test)
python split_dataset.py

# Bước 5: Huấn luyện toàn bộ 8 mô hình & tính Bootstrap CI
python train_models.py

# Bước 6: Hiệu chuẩn Conformal Quantile [P10, P90] & xuất báo cáo Slice Analysis
python analyze_and_calibrate.py
```

> **Mẹo chạy nhanh 1 dòng lệnh**:
> ```powershell
> python filter_hanoi.py; python handle_missing.py; python remove_duplicates.py; python clean_outliers.py; python split_dataset.py; python train_models.py; python analyze_and_calibrate.py
> ```

---

## 7. Hướng Dẫn Định Giá Thực Tế (Inference CLI & Rich Output)

Hệ thống hỗ trợ giao diện Terminal định dạng bảng màu Rich trực quan, hiển thị chi tiết khoảng tin cậy và giải thích TreeSHAP:

### 1. Chạy Demo Mẫu Trên 4 Bất Động Sản Thực Tế
```powershell
python predict_valuation.py --demo
```

### 2. Định Giá 1 Căn Nhà Trực Tiếp Qua Dòng Lệnh
```powershell
python predict_valuation.py `
  --type "Nhà" `
  --district "Cầu Giấy" `
  --ward "Yên Hòa" `
  --street "Trung Kính" `
  --area 55.0 `
  --floors 5 `
  --frontage 4.2 `
  --road-width 5.0 `
  --bedrooms 4 `
  --bathrooms 4 `
  --house-direction "Đông Nam" `
  --name "Bán nhà ngõ ô tô tránh Trung Kính 55m2 5 tầng thang máy mới kinh doanh" `
  --description "Nhà đẹp dân xây 5 tầng thang máy, ngõ thông ô tô tránh, sẵn sổ đỏ giao dịch."
```

### 3. Định Giá Hàng Loạt Từ File JSON
```powershell
python predict_valuation.py --input-json data_input.json
```

---

## 8. Kiểm Thử Tự Động (Automated Testing)

Dự án tích hợp bộ kiểm thử tự động toàn diện bằng `pytest`, kiểm tra độ chính xác, an toàn rò rỉ dữ liệu và tính bất biến:

```powershell
pytest -v
```

**Kết quả kiểm thử:**
* `tests/test_raw_immutability.py`: Xác thực dữ liệu `Raw` được bảo vệ tuyệt đối, không bị sửa đổi.
* `tests/test_cleaning.py`: Xác thực quy tắc regex, chuẩn hóa hướng, khử trùng lặp và tính toàn vẹn thuộc tính vật lý.
* `tests/test_no_leakage.py`: Xác thực tính đơn điệu của thời gian chia tập, độ giao nhau bằng rỗng giữa các nhóm BĐS, cơ chế Out-Of-Fold và việc không rò rỉ từ khóa giá tiền vào đặc trưng.
* `tests/test_models_and_inference.py`: Xác thực suy diễn điểm, kiểm tra độ phủ khoảng tin cậy không bị cắt chéo và tính hợp lệ của vector TreeSHAP.

```text
============================= 13 passed in 23.95s =============================
```

---

## 👨‍💻 Tác Giả & Bản Quyền
* Dự án được xây dựng phục vụ nghiên cứu và ứng dụng thực tế trong lĩnh vực Định giá Bất động sản Tự động (AVM) tại Việt Nam.
* Giấy phép: MIT License.
