# 🏡 Hanoi Real Estate Automated Valuation Model (AVM)

[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.14-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%7C%20HistGBR%20%7C%20Ridge-green.svg)](https://lightgbm.readthedocs.io/)
[![Zero Data Leakage](https://img.shields.io/badge/Data%20Leakage-Guaranteed%20Zero-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/pytest-13%20passed%20(100%25)-success.svg)](https://docs.pytest.org/)
[![Uncertainty](https://img.shields.io/badge/Quantile-80%25%20Mondrian%20Conformal-orange.svg)]()

Hệ thống **Định giá Bất động sản Tự động (Automated Valuation Model - AVM)** cấp độ sản xuất dành riêng cho hai phân khúc nhà ở lớn nhất tại **Hà Nội**: **Nhà** (nhà riêng, nhà mặt phố, phân lô, tòa CHDV) và **Căn hộ chung cư**.

Hệ thống dự đoán:
1. **Giá trị giao dịch hợp lý (Point Valuation)**: Tính bằng **VNĐ**, **Tỷ VNĐ** và đơn giá **Triệu VNĐ/m²**.
2. **Khoảng tin cậy định giá 80% (`[P10, P90]`)**: Hiệu chuẩn bằng **Mondrian Split Conformalized Quantile Regression (CQR)**, đảm bảo độ phủ thực tế và không cắt chéo phân vị.
3. **Giải thích cục bộ (TreeSHAP Explainer)**: Bóc tách Top 5 yếu tố cốt lõi (vị trí, phân tầng ngõ/phố, kết cấu, tiện ích) thúc đẩy định giá từng căn nhà.

---

## 📑 Mục Lục
- [1. Điểm Nhấn Kiến Trúc & Kỹ Thuật](#1-điểm-nhấn-kiến-trúc--kỹ-thuật)
- [2. Kiến Trúc Luồng Dữ Liệu (Medallion Pipeline)](#2-kiến-trúc-luồng-dữ-liệu-medallion-pipeline)
- [3. So Sánh Hiệu Năng Mô Hình (Benchmark Results)](#3-so-sánh-hiệu-năng-mô-hình-benchmark-results)
- [4. Định Lượng Bất Định (Mondrian Conformal Prediction)](#4-định-lượng-bất-định-mondrian-conformal-prediction)
- [5. Cấu Trúc Thư Mục Dự Án](#5-cấu-trúc-thư-mục-dự-án)
- [6. Hướng Dẫn Cài Đặt & Chạy Dự Án](#6-hướng-dẫn-cài-đặt--chạy-dự-án)
- [7. Hướng Dẫn Định Giá (CLI & Inference)](#7-hướng-dẫn-định-giá-cli--inference)
- [8. Kiểm Thử Tự Động (Automated Testing)](#8-kiểm-thử-tự-động-automated-testing)

---

## 1. Điểm Nhấn Kiến Trúc & Kỹ Thuật

* **Tuân thủ Nghiêm ngặt Zero Data Leakage**:
  * **Purged Chronological Walk-Forward Split**: Dữ liệu chia tăng dần theo dòng thời gian (`Train 70%` $\to$ `Val 15%` $\to$ `Test 15%`). Bất kỳ tin đăng nào trong tập Val/Test trùng nhóm tài sản (`property_group_id`) với tập trước đều bị loại bỏ triệt để (**Cross-split Group Purging**).
  * **Stateless Price-Redacted NLP**: Tự động nhận diện và xóa sạch mọi ký tự tiền tệ (`tỷ`, `triệu`, `tr`, `đ`, ...) trước khi trích xuất số tầng, số phòng ngủ, mặt tiền, lộ giới ngõ từ văn bản để triệt tiêu hoàn toàn rò rỉ nhãn mục tiêu.
  * **Hierarchical Out-Of-Fold (OOF) Target Encoding**: Mã hóa biến vị trí 5 cấp (Thành phố $\to$ Loại hình $\to$ Quận $\to$ Phường $\to$ Đường/Dự án) kèm làm mịn Empirical Bayes. Sử dụng 5-Fold OOF trên tập Train để không bản ghi huấn luyện nào nhìn thấy giá trị mục tiêu của chính nó.
* **Mô Hình Hồi Quy Phân Tầng Chuyên Biệt (Multi-Tier & Segmented GBDT)**:
  * Kết hợp mô hình chuyên gia cho Nhà đất (`House Expert`), mô hình chuyên gia cho Căn hộ (`Apartment Expert`) và mô hình toàn cục (`Unified LightGBM`) tối ưu hàm mất mát **Huber Loss** trên thang đo $\log(\text{price} / \text{area})$.
  * Tinh chỉnh hệ số tỷ lệ nhân vi phân $\alpha$ (Multiplicative Scale Factor) trên tập Validation để cân bằng tối ưu sai số phần trăm thực tế (`MAPE/MdAPE`).
* **Định Lượng Độ Bất Định Đáng Tin Cậy (Mondrian Conformal Quantile Regression)**:
  * Huấn luyện 2 mô hình LightGBM phân vị tại $\alpha = 0.10$ và $\alpha = 0.90$.
  * Áp dụng kỹ thuật hiệu chuẩn Mondrian Split Conformal trên tập Validation theo từng loại hình để đảm bảo khoảng biến thiên `[P10, P90]` phản ánh đúng mức độ biến động thị trường.
* **Giải Thích Mô Hình Minh Bạch (Explainable AI - XAI)**:
  * Tích hợp thuật toán **TreeSHAP** (`pred_contrib=True`) giải thích chính xác đóng góp của từng đặc tính cho từng lượt định giá độc lập.

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

## 3. So Sánh Hiệu Năng Mô Hình (Benchmark Results)

Đánh giá trung thực trên tập **Test Holdout** hoàn toàn chưa từng thấy (`11,498` bất động sản thực tế tại Hà Nội):

| Tầng Mô Hình | Mô Tả Thuật Toán | MAPE (%) | MdAPE (%) | PPE15 (%) | PPE20 (%) | $R^2_{\log}$ | $R^2_{\text{price}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Baseline** | Trung vị đơn giá theo (Loại, Quận) | 25.49% | 19.08% | 41.33% | 52.10% | 0.796 | 0.612 |
| **Tier 2: Linear** | Log-Linear Ridge ($L_2$ Regularization) | 18.11% | 12.90% | 56.10% | 68.01% | 0.894 | 0.781 |
| **Tier 3: HistGBR** | Scikit-Learn HistGBR (L1 Loss) | 15.78% | 10.80% | 63.32% | 74.49% | 0.913 | 0.830 |
| **Tier 4: Unified** | Global LightGBM (Huber Loss) | 15.21% | 10.47% | 64.34% | 75.04% | 0.920 | 0.847 |
| **Tier 5: Segmented**| Specialist Nhà & Chung cư LightGBM | 14.96% | 10.22% | 65.24% | 75.71% | 0.921 | 0.845 |
| **Tier 6: Champion** | **Ensemble Blend (Segmented + Unified)**| **14.95%**| **10.21%**| **65.22%**| **75.68%**| **0.921** | **0.846** |

> **Ý nghĩa chỉ số**:
> * **PPE20 = 75.68%**: Hơn **75.6%** các căn nhà được định giá chính xác trong biên độ sai số $\le 20\%$ so với giá trị thị trường.
> * **MdAPE = 10.21%**: Sai số phần trăm trung vị đạt xấp xỉ **10%** (đối với phân khúc Chung cư, MdAPE đạt tới **6.65%** và PPE20 đạt **90.57%**).
> * $R^2_{\log} = 0.921$: Mô hình giải thích được hơn **92.1%** phương sai giá trị nhà đất trên toàn Hà Nội.

---

## 4. Định Lượng Bất Định (Mondrian Conformal Prediction)

* **Mục tiêu độ phủ**: 80% (`target_coverage = 0.80`).
* **Độ phủ thực tế trên tập Test (`PICP_80`)**: **78.67%** (riêng phân khúc Nhà đạt **79.91%**).
* **Kiểm định tính đơn điệu**: **100%** không vi phạm cắt chéo phân vị (`pred_low <= pred_price <= pred_high`).
* **Độ rộng khoảng giá trung vị**: `46.93%`, phản ánh trung thực biên độ thương lượng tự nhiên của thị trường bất động sản.

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
│   ├── modeling.py                     # Huấn luyện 6 tầng mô hình, Bootstrap CI
│   ├── intervals_and_analysis.py       # Mondrian Conformal CQR, Slice Error Analysis
│   └── splitting.py                    # Purged chronological walk-forward split
├── tests/                              # Bộ kiểm thử tự động pytest
│   ├── test_cleaning.py                # Kiểm thử chuẩn hóa, regex, khử trùng
│   ├── test_raw_immutability.py        # Kiểm thử tính bất biến của dữ liệu Raw
│   ├── test_no_leakage.py              # Kiểm thử chống rò rỉ dữ liệu (Anti-leakage guards)
│   └── test_models_and_inference.py    # Kiểm thử suy diễn điểm, khoảng Conformal & TreeSHAP
├── analyze_and_calibrate.py            # Script hiệu chuẩn Conformal Quantile & Slice Report
├── clean_outliers.py                   # Script thực thi làm sạch Giai đoạn 3
├── HUONG_DAN_DU_AN_TUNG_BUOC.md        # Tài liệu hướng dẫn chi tiết từng bước
├── inspect_data.py / filter_hanoi.py   # Script thăm dò & lọc dữ liệu ban đầu
├── check_missing.py / handle_missing.py# Script kiểm toán & điền khuyết cơ bản
├── check_duplicates.py / remove_duplicates.py # Script kiểm tra & lọc trùng lặp ban đầu
├── predict_valuation.py                # CLI & Module Định giá Thực tế + TreeSHAP Explainer
├── split_dataset.py                    # Script phân chia dữ liệu Chronological Purged
├── train_models.py                     # Script huấn luyện & so sánh benchmark mô hình
└── requirements.txt                    # Danh sách thư viện phụ thuộc
```

---

## 6. Hướng Dẫn Cài Đặt & Chạy Dự Án

### Yêu Cầu Môi Trường
* Python 3.10 trở lên (khuyến nghị Python 3.11 - 3.14).

### Cài Đặt
```bash
# 1. Clone repository
git clone https://github.com/Banam144/PROPERTY-VALUATION.git
cd PROPERTY-VALUATION

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Trên Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### Chạy Toàn Bộ Pipeline Huấn Luyện (End-to-End)
Nếu bạn có tập dữ liệu gốc `data/raw/shard_0000.parquet`, bạn có thể tái lập toàn bộ pipeline:

```bash
# Bước 1: Lọc dữ liệu Hà Nội
python filter_hanoi.py

# Bước 2: Xử lý Missing Values & Khử trùng lặp sơ bộ
python handle_missing.py
python remove_duplicates.py

# Bước 3: Làm sạch chuyên sâu, chuẩn hóa thực thể & khử broker repost
python clean_outliers.py

# Bước 4: Chia tập Purged Chronological Split (Train/Val/Test)
python split_dataset.py

# Bước 5: Huấn luyện 6 tầng mô hình & tính Bootstrap CI
python train_models.py

# Bước 6: Hiệu chuẩn Conformal Quantile [P10, P90] & xuất báo cáo Slice Analysis
python analyze_and_calibrate.py
```

---

## 7. Hướng Dẫn Định Giá (CLI & Inference)

Module `predict_valuation.py` hỗ trợ định giá tức thì cho mọi căn nhà/căn hộ tại Hà Nội kèm giải thích đóng góp đặc trưng TreeSHAP.

### 1. Chạy Demo Mẫu Trên Dữ Liệu Thực Tế
```bash
python predict_valuation.py --demo
```

### 2. Định Giá Tương Tác Trực Tiếp Qua CLI
```bash
python predict_valuation.py \
  --type "Căn hộ chung cư" \
  --district "Cầu Giấy" \
  --ward "Trung Hòa" \
  --street "Trần Duy Hưng" \
  --project "Vinhomes D'Capitale" \
  --area 85.0 \
  --bedrooms 3 \
  --bathrooms 2 \
  --balcony "Đông Nam" \
  --name "Căn hộ 3PN góc view hồ full đồ cao cấp sổ đỏ" \
  --desc "Căn góc 3 phòng ngủ thoáng mát, full nội thất 5 sao, sẵn sổ đỏ sang tên."
```

### 3. Định Giá Hàng Loạt Từ File JSON
```bash
python predict_valuation.py --input-json data_can_dinh_gia.json --output-json ket_qua_dinh_gia.json
```

### Minh Họa Kết Quả Đầu Ra:
```json
{
  "input_summary": {
    "property_type_name": "Căn hộ chung cư",
    "district_name": "Cầu Giấy",
    "ward_name": "Trung Hòa",
    "project_name": "Vinhomes D'Capitale",
    "area_m2": 85.0
  },
  "valuation": {
    "fair_price_vnd": 8838381806,
    "fair_price_ty": 8.838,
    "unit_price_million_per_m2": 103.98,
    "conformal_80_low_ty": 7.935,
    "conformal_80_high_ty": 9.942,
    "interval_width_pct": 22.71
  },
  "top_treeshap_drivers": [
    {
      "feature": "te_project_log_ppm2",
      "approx_unit_price_effect_pct": 28.45,
      "log_contribution": 0.2504
    },
    {
      "feature": "is_lo_goc",
      "approx_unit_price_effect_pct": 4.12,
      "log_contribution": 0.0404
    },
    {
      "feature": "has_full_noi_that",
      "approx_unit_price_effect_pct": 3.25,
      "log_contribution": 0.0320
    }
  ]
}
```

---

## 8. Kiểm Thử Tự Động (Automated Testing)

Dự án tích hợp bộ kiểm thử tự động toàn diện bằng `pytest`, kiểm tra độ chính xác, an toàn rò rỉ dữ liệu và tính bất biến:

```bash
pytest -v
```

**Kết quả kiểm thử:**
* `tests/test_raw_immutability.py`: Xác thực dữ liệu `Raw` được bảo vệ tuyệt đối, không bị sửa đổi.
* `tests/test_cleaning.py`: Xác thực quy tắc regex, chuẩn hóa hướng, khử trùng lặp và tính toàn vẹn thuộc tính vật lý.
* `tests/test_no_leakage.py`: Xác thực tính đơn điệu của thời gian chia tập, độ giao nhau bằng rỗng giữa các nhóm BĐS, cơ chế Out-Of-Fold và việc không rò rỉ từ khóa giá tiền vào đặc trưng.
* `tests/test_models_and_inference.py`: Xác thực suy diễn điểm, kiểm tra độ phủ khoảng tin cậy không bị cắt chéo và tính hợp lệ của vector TreeSHAP.

```text
============================== 13 passed in 14.82s ==============================
```

---

## 👨‍💻 Tác Giả & Bản Quyền
* Dự án được xây dựng phục vụ nghiên cứu và ứng dụng thực tế trong lĩnh vực Định giá Bất động sản Tự động (AVM) tại Việt Nam.
* Giấy phép: MIT License.
