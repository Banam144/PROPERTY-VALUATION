# HƯỚNG DẪN DỰ ÁN TỪNG BƯỚC: ĐỊNH GIÁ BẤT ĐỘNG SẢN HÀ NỘI (`PROPERTY-VALUATION`)

> **Mục tiêu dự án**: Xây dựng hệ thống Định giá Bất động sản tự động (**Automated Valuation Model - AVM**) cho hai phân khúc nhà ở phổ biến nhất tại **Hà Nội** là **Nhà** (nhà riêng, nhà mặt phố, nhà phân lô, tòa CHDV) và **Căn hộ chung cư**, dự đoán **giá trị giao dịch hợp lý (VNĐ / Tỷ VNĐ)** cùng **khoảng tin cậy định giá 80% (`[P10, P90]`)** từ đặc điểm vị trí, diện tích, kết cấu vật lý và mô tả thực tế — tuân thủ nghiêm ngặt kiến trúc dữ liệu **Medallion (`Raw → Silver → Gold`)**, **Zero Data Leakage** và kiểm thử tự động toàn diện bằng `pytest`.

---

## MỤC LỤC & TRẠNG THÁI TỔNG QUAN DỰ ÁN

| Giai đoạn        | Mức ưu tiên | Hạng mục công việc                                                                                   | File chính                                                  |    Trạng thái     | Kết quả kiểm chứng thực tế                                                                                                                                                  |
| :--------------- | :---------: | :--------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- | :---------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Giai đoạn 0**  |  Nền tảng   | Khởi tạo môi trường `.venv` & Quản lý thư viện                                                       | `.venv/`, `requirements.txt`                                | **Đã hoàn thành** | Python `3.14.4`, `pandas 3.1.1`, `numpy 2.5.5`, `pyarrow 26.0.1`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `scipy 1.18.1`, `joblib 1.6.0`, `pytest 9.2.0`                    |
| **Giai đoạn 1**  |    Đã có    | Khám phá dữ liệu gốc & Lọc BĐS nhà ở Hà Nội (`Raw Layer`)                                            | `inspect_data.py`, `filter_hanoi.py`                        | **Đã hoàn thành** | `350,075` dòng toàn quốc → `121,629` dòng Hà Nội → `93,940` dòng (`Nhà`: `65,291`, `Căn hộ chung cư`: `28,649`)                                                             |
| **Giai đoạn 2A** |    Đã có    | Phân tích & Xử lý Missing Values cơ bản (`Silver Layer`)                                             | `check_missing.py`, `handle_missing.py`                     | **Đã hoàn thành** | Loại `3,002` dòng thiếu `price` (`3.20%`) → còn `90,938` dòng, `25` cột (thêm 6 cờ `*_missing`)                                                                             |
| **Giai đoạn 2B** |    Đã có    | Phát hiện & Khử trùng lặp cơ bản (`Silver Layer`)                                                    | `check_duplicates.py`, `remove_duplicates.py`               | **Đã hoàn thành** | Phát hiện `3,055` nhóm nghi trùng, loại `865` bản ghi trùng khớp khóa → còn `90,073` dòng (`hanoi_deduplicated.parquet`)                                                    |
| **Giai đoạn 3**  |   **P0**    | Làm sạch chuyên sâu: Chuẩn hóa thực thể, Xử lý ngoại lai/sai đơn vị & Khử Near-Duplicate Repost      | `src/data_cleaning.py`, `clean_outliers.py`                 | **Đã hoàn thành** | Chuẩn hóa hướng (`13→9`), loại `444` ngoại lai/tin thuê, sanitize `171` ô vật lý, khử `11,976` repost → còn `77,653` dòng sạch (`76,493` nhóm tài sản); `6/6` pytest PASSED |
| **Giai đoạn 4**  |   **P0**    | Chia tập Chronological Group-Safe (`Gold Layer`) & Pipeline Đặc trưng Chống Data Leakage             | `src/splitting.py`, `src/features.py`, `split_dataset.py`   | **Đã hoàn thành** | `Train` (`54,357`), `Val` (`11,480`, purge `168`), `Test` (`11,498`, purge `150`); khôi phục `27,056` số tầng từ văn bản; `110` đặc trưng sạch; `9/9` pytest PASSED         |
| **Giai đoạn 5**  |   **P0**    | Huấn luyện Hệ thống Mô hình Đa Tầng (Baseline → Ridge → HistGBR → LightGBM) & Đánh giá Bootstrap CI  | `src/modeling.py`, `train_models.py`                        | **Đã hoàn thành** | Test Holdout (`11,498` BĐS): `MAPE = 14.95%` (giảm từ `25.49%` baseline), `MdAPE = 10.21%`, `PPE20 = 75.68%`, $R^2_{\log} = 0.921$, $R^2_{\text{price}} = 0.846$            |
| **Giai đoạn 6**  |   **P1**    | Dự đoán Khoảng giá Conformal Quantile (`[P10, P50, P90]`), Slice Error Analysis & Giải thích Mô hình | `src/intervals_and_analysis.py`, `analyze_and_calibrate.py` | **Đã hoàn thành** | Độ phủ `PICP = 78.67%` (riêng `Nhà` đạt `79.91%`, `0%` cắt chéo); Chung cư đạt `MdAPE = 6.65%`, `PPE20 = 90.57%`; xuất bảng Gain 110 đặc trưng                              |
| **Giai đoạn 7**  |   **P1**    | Module/CLI Định giá Thực tế kèm Giải thích TreeSHAP & Bộ Kiểm thử Tự động `pytest` Toàn diện         | `predict_valuation.py`, `tests/`                            | **Đã hoàn thành** | Suy diễn CLI/JSON + Top 5 đóng góp TreeSHAP từng căn nhà; toàn bộ **`13/13` bài kiểm thử `pytest` PASSED (0 cảnh báo)**                                                     |

---

## PHẦN I: KIỂM TOÁN CHI TIẾT TRẠNG THÁI HIỆN TẠI (GIAI ĐOẠN 1 & 2A–2B)

> **Lưu ý kiểm toán ban đầu**: Khi tiếp nhận dự án, kho mã nguồn có 6 script rời ở thư mục gốc (`inspect_data.py`, `filter_hanoi.py`, `check_missing.py`, `handle_missing.py`, `check_duplicates.py`, `remove_duplicates.py`), 4 tệp Parquet trong `data/raw/` và `data/silver/`, cùng 2 báo cáo CSV trong `reports/`. Tệp `HUONG_DAN_DU_AN_TUNG_BUOC.md` chưa tồn tại trước đó và được khởi tạo tại đây để ghi nhận trung thực toàn bộ trạng thái đã làm cũng như cập nhật chi tiết sau từng giai đoạn tiếp theo.

### 1. Kiến trúc Luồng Dữ liệu Hiện có

```
data/raw/shard_0000.parquet (350,075 dòng × 19 cột — BẤT BIẾN / READ-ONLY)
   │
   ├──► [inspect_data.py] ──► Thống kê tổng quan toàn quốc & Hà Nội
   │
   └──► [filter_hanoi.py]
           │  Lọc province_name == "Hà Nội" & property_type_name ∈ {"Nhà", "Căn hộ chung cư"}
           ▼
data/raw/hanoi_housing_raw.parquet (93,940 dòng × 19 cột — BẤT BIẾN / READ-ONLY)
   │
   ├──► [check_missing.py] ──► Báo cáo tỷ lệ khuyết thiếu (cả NaN lẫn chuỗi rỗng "")
   │
   └──► [handle_missing.py]
           │  Chuẩn hóa chuỗi rỗng → NA, bỏ dòng thiếu price/area (-3,002 dòng),
           │  điền "Không rõ" cho categorical, tạo 6 cờ *_missing (chưa impute số!)
           ▼
data/silver/hanoi_missing_handled.parquet (90,938 dòng × 25 cột)
   │
   ├──► [check_duplicates.py] ──► reports/duplicate_candidates.csv (3,055 nhóm nghi trùng)
   │
   └──► [remove_duplicates.py]
           │  Chuẩn hóa tiêu đề/mô tả, sắp xếp published_at tăng dần, giữ bản ghi đăng sớm nhất
           ▼
data/silver/hanoi_deduplicated.parquet (90,073 dòng × 25 cột) + reports/removed_duplicates.csv (865 dòng)
```

---

### 2. Giải thích Chi tiết Từng File Đã Hoàn Thành

#### 2.1. File `inspect_data.py` — Khám phá Dữ liệu Thô Toàn quốc

- **Mục đích**: Đọc phân mảnh dữ liệu gốc `data/raw/shard_0000.parquet` để nắm bắt quy mô, danh sách 19 trường thông tin, phân bố địa lý theo tỉnh/thành phố và cơ cấu loại hình BĐS tại Hà Nội trước khi quyết định phạm vi bài toán.
- **Luồng xử lý & Cú pháp Python/Pandas**:
    - `pd.read_parquet("data/raw/shard_0000.parquet")`: Đọc định dạng cột nén **Apache Parquet** thông qua engine `pyarrow`, giữ nguyên kiểu dữ liệu gốc và nhanh hơn gấp nhiều lần so với CSV trên tập 350 nghìn dòng.
    - `df["province_name"].value_counts().head(10)`: Thống kê tần suất giảm dần để xác định thị trường lớn nhất.
    - `df["province_name"].astype(str).str.strip() == "Hà Nội"`: Cắt khoảng trắng đầu/cuối trước khi so khớp chuỗi để tránh bỏ sót bản ghi có ký tự trắng thừa.
- **Kết quả chạy thực tế**:
    - Tổng số bản ghi gốc: **`350,075` dòng × `19` cột** (`name`, `description`, `property_type_name`, `province_name`, `district_name`, `ward_name`, `street_name`, `project_name`, `price`, `area`, `floor_count`, `frontage_width`, `house_depth`, `road_width`, `bedroom_count`, `bathroom_count`, `house_direction`, `balcony_direction`, `published_at`).
    - Top 3 tỉnh/thành nhiều tin đăng nhất: **Hà Nội (`121,629` tin)**, **Hồ Chí Minh (`119,308` tin)**, **Đà Nẵng (`19,117` tin)**.
    - Cơ cấu 5 loại hình tại Hà Nội (`121,629` tin):
        - `Nhà`: `65,291` tin (`53.68%`)
        - `Căn hộ chung cư`: `28,649` tin (`23.55%`)
        - `Đất`: `18,121` tin (`14.90%`)
        - `Biệt thự/Nhà liền kề`: `8,691` tin (`7.15%`)
        - `Shophouse`: `877` tin (`0.72%`)

#### 2.2. File `filter_hanoi.py` — Trích lọc Tập Dữ liệu Nhà ở Hà Nội

- **Mục đích**: Tách riêng thị trường Hà Nội và chỉ giữ 2 loại hình nhà ở dân dụng có cơ chế hình thành giá rõ ràng nhất là `Nhà` và `Căn hộ chung cư` (loại bỏ `Đất` nền vốn không có kết cấu xây dựng và `Shophouse`/`Biệt thự` có đặc thù thương mại riêng biệt).
- **Chi tiết kỹ thuật & Ghi chú kiểm toán**:
    - Dùng `fillna("").str.strip()` trên cả `province_name` và `property_type_name` rồi lọc `.isin(["Nhà", "Căn hộ chung cư"])`.
    - _Lưu ý nhỏ phát hiện khi đọc code_: Ở dòng 26, câu lệnh `housing.reset_index(drop=True)` không gán lại vào biến `housing = ...` (vì mặc định `inplace=False`). Tuy nhiên ở dòng 35, `housing.to_parquet(output_path, index=False)` đã bỏ qua index khi ghi xuống đĩa nên file kết quả `data/raw/hanoi_housing_raw.parquet` hoàn toàn sạch và không bị thêm cột `__index_level_0__`. Theo nguyên tắc **bất biến dữ liệu Raw**, chúng ta giữ nguyên tuyệt đối tệp này.
- **Kết quả thực tế**: Xuất `data/raw/hanoi_housing_raw.parquet` gồm **`93,940` dòng × `19` cột** (`65,291` Nhà + `28,649` Căn hộ chung cư).

#### 2.3. File `check_missing.py` — Kiểm toán Giá trị Khuyết thiếu (Missing Values)

- **Mục đích**: Đo lường chính xác tỷ lệ khuyết thiếu trên từng cột của `93,940` tin đăng Hà Nội.
- **Kiến thức Python/ML quan trọng**:
    - Dữ liệu thu thập từ web (web scraping) thường chứa **giá trị thiếu ẩn (disguised missing values)** dưới dạng chuỗi rỗng `""` hoặc chỉ toàn dấu cách `"   "`. Nếu chỉ gọi `df.isna().sum()` thông thường sẽ bỏ sót chúng.
    - Script kết hợp mặt nạ logic `missing_mask[col] = missing_mask[col] | empty_mask` với `df[col].astype("string").str.strip().eq("").fillna(False)` để bắt trọn cả hai dạng thiếu.
- **Kết quả chạy thực tế trên `93,940` dòng**:
    - Các cột đầy đủ `100%` (`0%` thiếu): `area`, `district_name`, `province_name`, `property_type_name`, `published_at`.
    - Cột mục tiêu `price`: thiếu **`3,002` dòng (`3.20%`)** (thường là các tin để _"Giá thỏa thuận"_ trên website).
    - Các cột cấu trúc vật lý thiếu rất cao do người đăng bỏ trống ô nhập tùy chọn: `house_depth` (`98.75%`), `road_width` (`92.03%`), `balcony_direction` (`78.15%`), `floor_count` (`76.96%`), `house_direction` (`74.83%`), `project_name` (`72.59%`), `frontage_width` (`57.57%`), `bathroom_count` (`35.94%`), `bedroom_count` (`30.72%`), `street_name` (`26.91%`), `ward_name` (`12.82%`).

#### 2.4. File `handle_missing.py` — Xử lý Missing Values Không Gây Rò rỉ Dữ liệu

- **Mục đích**: Làm sạch sơ bộ giá trị thiếu và chuyển dữ liệu từ `Raw` sang tầng `Silver` (`data/silver/hanoi_missing_handled.parquet`).
- **Luồng xử lý & Tư duy Chống Data Leakage**:
    1. Chuẩn hóa mọi chuỗi rỗng sau `.str.strip()` về `pd.NA`; ép kiểu số an toàn cho `price` và `area` bằng `pd.to_numeric(..., errors="coerce")`.
    2. **Loại bỏ dòng thiếu nhãn mục tiêu (`price`) hoặc `area`**: Trong học có giám sát (Supervised Regression), không thể huấn luyện hay đánh giá trung thực trên mẫu không có nhãn thật `price` (nếu tự điền `price` giả sẽ làm sai lệch phân phối mục tiêu). Đã loại chính xác **`3,002` dòng**, giữ lại **`90,938` dòng**.
    3. **Xử lý biến phân loại (`ward_name`, `street_name`, `project_name`, `house_direction`, `balcony_direction`)**: Điền nhãn tường minh `"Không rõ"`. Trong BĐS, việc thiếu `project_name` không phải lỗi ngẫu nhiên (**MNAR - Missing Not At Random**): nhà thổ cư trong dân hầu hết không thuộc dự án nào (`~95%` thiếu), trong khi căn hộ chung cư thường có tên dự án! Nhãn `"Không rõ"` giúp mô hình giữ lại tín hiệu này.
    4. **Xử lý 6 cột số (`floor_count`, `frontage_width`, `house_depth`, `road_width`, `bedroom_count`, `bathroom_count`)**: Tạo 6 cột cờ nhị phân `*_missing` kiểu `int8` (`0` hoặc `1`) và **CỐ Ý GIỮ NGUYÊN `NaN`** thay vì điền trung vị (`median`) trên toàn tập! Lý do: Nếu tính `median` trên toàn bộ 90 nghìn dòng trước khi chia `Train/Val/Test`, thông tin thống kê từ tập `Test` sẽ rò rỉ sang tập `Train` (**Preprocessing Data Leakage**). Việc điền giá trị khuyết thiếu cho mô hình tuyến tính bắt buộc chỉ được `fit` trên `X_train` ở Giai đoạn 4–5 (còn LightGBM/HistGBR xử lý trực tiếp `NaN` kết hợp cờ `*_missing`).
- **Kết quả thực tế**: Xuất `data/silver/hanoi_missing_handled.parquet` gồm **`90,938` dòng × `25` cột**.

#### 2.5. File `check_duplicates.py` & `remove_duplicates.py` — Kiểm tra và Loại bỏ Tin Trùng Lặp Cơ bản

- **Mục đích**: Xử lý hiện tượng môi giới đăng đi đăng lại cùng một tin rao ở nhiều thời điểm khác nhau (`published_at` khác nhau).
- **Cơ chế hoạt động**:
    - Hàm `normalize_text(series)`: Chuyển chữ thường Unicode bằng `.str.casefold()` (mạnh và chuẩn xác hơn `.lower()` cho đa ngôn ngữ), gộp nhiều khoảng trắng/xuống dòng liên tiếp `r"\s+"` thành 1 dấu cách đơn và `.str.strip()`.
    - Chỉ xét tự động các tin có tiêu đề chuẩn hóa `>= 20` ký tự và mô tả `>= 40` ký tự (`valid_text` đạt `90,020 / 90,938` tin) để tránh gộp nhầm các tin mô tả quá ngắn kiểu _"Bán nhà Cầu Giấy"_.
    - Sắp xếp tăng dần theo thời gian `published_at` với `kind="stable"` (thuật toán Timsort/Mergesort bảo toàn thứ tự tương đối khi bằng nhau), sau đó gọi `.duplicated(subset=compare_cols, keep="first")` để **giữ lại bản ghi xuất hiện sớm nhất trên thị trường** và loại bỏ các bản sao đăng lại về sau.
- **Kết quả thực tế**:
    - Loại bỏ **`865` bản ghi trùng**, xuất danh sách đối chiếu ra `reports/removed_duplicates.csv`.
    - Lưu tập dữ liệu `data/silver/hanoi_deduplicated.parquet` gồm **`90,073` dòng × `25` cột** (`63,690` Nhà và `26,383` Căn hộ chung cư trên toàn bộ `30` Quận/Huyện/Thị xã của Hà Nội, thời gian đăng từ `01/06/2025` đến `30/07/2025`).

---

### 3. Các Vấn Đề Nghiêm Trọng Phát Hiện Qua Kiểm Toán Dữ Liệu Sâu (Lý Do Cần Giai Đoạn 3 → 7)

Khi chạy kiểm toán định lượng trực tiếp trên `data/silver/hanoi_deduplicated.parquet` (`90,073` dòng), chúng ta phát hiện **5 vấn đề cốt lõi** bắt buộc phải giải quyết ở các giai đoạn tiếp theo:

1. **Nhiễu đơn vị giá 1000x, nhập nhầm đơn giá/m² vào tổng giá & tin cho thuê lọt lưới**:
    - Có `274` dòng có đơn giá `< 15 triệu/m²` (thậm chí `min = 0.05 triệu/m²`), bao gồm: người bán nhập thiếu 3 số 0 (`price = 25,620,000` cho căn nhà ghi rõ _"giá 25.5 tỷ"_ trong tiêu đề), nhập nhầm đơn giá `66 triệu/m²` vào ô tổng giá `price = 66,000,000`, hoặc tin cho thuê phòng/căn hộ theo tháng.
    - Ở chiều ngược lại, `max(price) = 595,000 tỷ VNĐ` (`5.95e14`) và `72` dòng có đơn giá `> 1,500 triệu/m²` (lên tới `27,045,454 triệu/m²`) do người bán nhân đôi `tổng giá × diện tích` (ví dụ căn nhà 62m² giá `9.35 tỷ` bị nhập thành `579.7 tỷ = 9.35 × 62`!) hoặc nhập nhầm `frontage_width = 3.7m` vào ô `area = 3.7m²`!
    - Nếu đưa trực tiếp vào mô hình hồi quy, các điểm ngoại lai sai số hàng nghìn lần này sẽ làm nổ phương sai (exploding variance) và làm sai lệch hoàn toàn cả quá trình huấn luyện lẫn chỉ số RMSE/MAPE!
2. **Phân mảnh nhãn danh mục & lỗi parse thời gian Pandas 3**:
    - Cột `house_direction` và `balcony_direction` có 13 giá trị thay vì 9, do cùng một hướng nhưng bị tách thành `"Đông Nam"` (`4,028` dòng) và `"Đông - Nam"` (`1,291` dòng), tương tự cho `Tây Bắc`, `Đông Bắc`, `Tây Nam`.
    - Cột `street_name` có `314` bản ghi dính tiền tố `"Đường "` / `"Phố "` ở đầu trong khi phần còn lại chỉ lưu tên riêng (`"Thái Hà"`, `"Cầu Giấy"`).
    - Cột `published_at` có `104` bản ghi không có phần thập phân microsecond (`YYYY-MM-DDTHH:MM:SS`), khiến `pd.to_datetime()` mặc định của Pandas 3 ép thành `NaT` nếu không truyền `format="ISO8601"`.
3. **Các giá trị vật lý phi thực tế trong cột số**:
    - `frontage_width` lên tới `695m` (nhiều dòng quên dấu chấm thập phân: nhập `45` thay vì `4.5m` cho nhà `50m²`), `bedroom_count` và `bathroom_count` lên tới `96` phòng, `road_width` lên tới `150m`.
4. **Còn sót hàng nghìn bản ghi Near-Duplicate / Broker Repost gây nguy cơ Data Leakage**:
    - Trong `remove_duplicates.py`, khóa so sánh yêu cầu trùng khớp đồng thời toàn bộ 16 cột gốc và toàn bộ `description`. Thực tế kiểm tra cho thấy có **`6,868` dòng** trùng khít `(property_type_name, district_name, ward_name, area, price, _name_norm)` nhưng chỉ khác nhau đúng chuỗi `[phone_number]` ở cuối mô tả hoặc khác một cột phụ `"Không rõ"`! Nếu không khử repost và không khóa nhóm tài sản (`property_group_id`) khi chia tập, cùng một căn nhà sẽ lọt vào cả `Train` lẫn `Test`!
5. **Nguy cơ Target Leakage nghiêm trọng trong văn bản (`name` & `description`)**:
    - Đa số tiêu đề tin rao ở Việt Nam viết thẳng giá bán vào câu chữ (ví dụ: _"Bán nhà Cầu Giấy 50m2 5 tầng giá 12.5 tỷ"_). Nếu dùng TF-IDF / Bag-of-Words hoặc Regex bừa bãi lên văn bản mà không loại bỏ triệt để thông tin giá tiền, mô hình sẽ "đọc đáp án" từ tiêu đề thay vì định giá bất động sản thực sự!

---

## PHẦN II: GIAI ĐOẠN 0 & GIAI ĐOẠN 3 (P0) — LÀM SẠCH CHUYÊN SÂU, CHUẨN HÓA THỰC THỂ & KHỬ NEAR-DUPLICATE REPOST

### 1. Giai đoạn 0: Chuẩn hóa Môi trường Phụ thuộc (`.venv` & `requirements.txt`)

- **Lý do thực hiện**: Tuân thủ chuẩn quản lý phụ thuộc cô lập theo dự án (không bao giờ cài `pip install` toàn cục ra hệ điều hành để tránh xung đột phiên bản ABI giữa các thư viện Cython/C++ như `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`).
- **Thư viện cài đặt & Vai trò trong dự án**:
    - `pandas==3.1.1` & `pyarrow==26.0.1`: Xử lý dữ liệu bảng hiệu năng cao với kiểu chuỗi `pyarrow` (`string`) và lưu trữ định dạng nén cột `.parquet`.
    - `numpy==2.5.5` & `scipy==1.18.1`: Tính toán đại số tuyến tính, biến đổi logarit và kiểm định phân phối / Bootstrap CI.
    - `scikit-learn==1.9.1`: Khung kiến trúc `BaseEstimator`, `TransformerMixin`, `Pipeline`, `KFold` Out-Of-Fold, `Ridge`, `HistGradientBoostingRegressor`.
    - `lightgbm==4.7.0`: Thuật toán Gradient Boosted Decision Trees (GBDT) theo chiều sâu lá (_leaf-wise_) hỗ trợ hàm mất mát hồi quy điểm (`L1`/`Huber`/`L2`) và hồi quy phân vị (`quantile` cho khoảng tin cậy `[P10, P50, P90]`).
    - `joblib==1.6.0`: Tuần tự hóa (serialization) toàn bộ pipeline tiền xử lý và mô hình đã huấn luyện xuống đĩa.
    - `pytest==9.2.0`: Khung kiểm thử tự động xác minh bất biến dữ liệu Raw, tính đúng đắn của hàm làm sạch, chống rò rỉ dữ liệu và chất lượng dự đoán.
- **Cách cài đặt / tái lập môi trường**:
    ```powershell
    .\.venv\Scripts\pip.exe install -r requirements.txt
    ```

---

### 2. Luồng Dữ liệu Giai đoạn 3 (`Silver Deduplicated → Silver Cleaned`)

```
data/silver/hanoi_deduplicated.parquet (90,073 dòng × 25 cột)
   │
   ▼  [src/data_cleaning.py :: run_cleaning_pipeline()]
   ├─ Bước 1: normalize_direction(), normalize_street_name(), parse_published_at()
   │          • Gom "Đông - Nam" → "Đông Nam" (13 nhãn → 9 nhãn chuẩn)
   │          • Gỡ tiền tố "Đường "/"Phố " dư thừa (bảo toàn đường số như "Đường 800A")
   │          • Chuẩn hóa 100% chuỗi published_at về chuẩn ISO-8601 UTC (0 lỗi NaT)
   │
   ├─ Bước 2: filter_domain_outliers()  [Quy tắc vật lý/thị trường ĐỘC LẬP TỪNG DÒNG — 0% Leakage]
   │          • Loại tin cho thuê bị phân loại nhầm & lỗi đơn vị 1000x (ppm2 < 15 tr/m²)
   │          • Tách ngưỡng riêng cho Căn hộ chung cư vs Nhà thổ cư
   │          • Tuyệt đối xóa biến tạm ppm2 trước khi trả về DataFrame sạch!
   │
   ├─ Bước 3: sanitize_physical_attributes()  [Bảo toàn dòng hợp lệ theo ML Best Practices]
   │          • Đưa các ô nhập sai vật lý (VD: mặt tiền 42m trên đất 40m², 96 phòng ngủ) về NaN
   │          • Đồng bộ lại 6 cột cờ *_missing = isna().astype("int8")
   │
   └─ Bước 4: deduplicate_reposts_and_assign_group_id()
              • Xóa [phone_number] & dấu câu, sắp xếp published_at tăng dần (giữ tin sớm nhất)
              • Loại bỏ 11,976 tin môi giới đăng lại (trùng khóa tiêu đề hoặc trùng mô tả >= 50 ký tự)
              • Băm SHA-1 tạo mã định danh tài sản `property_group_id` (không phụ thuộc giá!)
   │
   ├──► data/silver/hanoi_cleaned.parquet (77,653 dòng × 26 cột)
   ├──► reports/outlier_and_repost_report.csv (12,420 dòng bị loại kèm lý do chi tiết)
   └──► reports/cleaning_summary.json (Báo cáo kiểm toán định lượng JSON)
```

---

### 3. Giải thích Chi tiết Từng File & Từng Hàm trong Giai đoạn 3

#### 3.1. File [src/data_cleaning.py](file:///d:/PROPERTY-VALUATION/src/data_cleaning.py)

Đây là module lõi chứa toàn bộ hàm thuần khiết (_pure functions_) phục vụ làm sạch chuyên sâu tầng `Silver`:

1. **`normalize_direction(series: pd.Series) -> pd.Series`**:
    - **Vấn đề giải quyết**: Dữ liệu gộp từ nhiều nguồn web viết hướng nhà không đồng nhất (`"Đông - Nam"` vs `"Đông Nam"`, chữ hoa/thường lộn xộn).
    - **Cơ chế**: Dùng biểu thức chính quy `.str.replace(r"\s*-\s*", " ", regex=True)` xóa dấu gạch ngang giữa hai từ chỉ hướng, chuẩn hóa khoảng trắng, đối chiếu không phân biệt hoa thường (`.str.casefold()`) với tập 8 hướng la bàn chuẩn `VALID_DIRECTIONS`. Mọi giá trị rỗng hoặc sai chuẩn được ánh xạ về `"Không rõ"`.
    - **Kết quả thực tế**: Số nhãn duy nhất của cả `house_direction` và `balcony_direction` giảm từ **`13` xuống đúng `9`** (8 hướng + `"Không rõ"`), giúp hợp nhất hàng nghìn mẫu bị chia cắt trước đó.

2. **`normalize_street_name(series: pd.Series) -> pd.Series`**:
    - **Vấn đề giải quyết**: Đa số bản ghi lưu `"Thái Hà"`, `"Cầu Giấy"`, nhưng có `314` bản ghi lưu `"Phố Thái Hà"`, `"Đường Cầu Giấy"`. Nếu giữ nguyên, bộ mã hóa địa lý sẽ hiểu đây là 2 con đường khác nhau.
    - **Kỹ thuật Regex nâng cao (Negative Lookahead `(?![0-9])`)**:
        - Mẫu `r"^(?:Đường|Phố|đường|phố)\s+(?![0-9])"` chỉ cắt tiền tố `"Đường "` hoặc `"Phố "` khi ký tự theo sau **không phải chữ số**.
        - Nhờ vậy, `"Phố Thái Hà"` → `"Thái Hà"`, nhưng các tuyến đường mang tên số đặc trưng của Hà Nội như `"Đường 800A"`, `"Đường 70"`, `"Đường 32"` vẫn được bảo toàn nguyên vẹn (không bị biến thành chuỗi số vô nghĩa `"70"`!).

3. **`parse_published_at(series: pd.Series) -> pd.Series`**:
    - **Kiến thức Pandas 3.x**: Trong Pandas 3, `pd.to_datetime()` suy luận định dạng nghiêm ngặt từ mẫu đầu tiên (`%Y-%m-%dT%H:%M:%S.%f`). Khi gặp `104` bản ghi có giây tròn không chứa phần thập phân microsecond (`2025-06-26T02:00:20`), hàm mặc định sẽ ép thành `NaT`. Truyền tường minh `format="ISO8601"` và `utc=True` giúp đọc chính xác `100%` (`90,073 / 90,073`) mốc thời gian.

4. **`filter_domain_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]`**:
    - **Tại sao dùng Quy tắc Nghiệp vụ Bất biến (Domain Physical Bounds) thay vì `Mean ± 3*Std` hay `IQR` trên toàn bộ dữ liệu?**
        - **Kiến thức ML — Chống Data Leakage**: Nếu tính trung bình/phương sai hoặc phân vị 1%–99% trên toàn bộ bảng dữ liệu trước khi chia `Train/Val/Test`, ngưỡng cắt đã nhìn thấy phân phối của tập `Test`. Ngược lại, quy tắc kiểm tra tính hợp lệ vật lý và đơn vị tiền tệ trên từng dòng độc lập $f(x_i, y_i) \in \{0, 1\}$ là phép lọc xác định (_row-independent predicate_), hoàn toàn không rò rỉ thống kê giữa các mẫu!
        - Đồng thời, cơ chế hình thành giá của `Căn hộ chung cư` (đơn giá sàn căn hộ) và `Nhà` (đơn giá đất + tài sản gắn liền với đất) tại Hà Nội khác nhau một bậc độ lớn nên bắt buộc phải tách ngưỡng theo `property_type_name`:
            - **Tin cho thuê lọt lưới / Lỗi đơn vị 1000x chung**: Tiêu đề bắt đầu bằng `"cho thuê"`, `"cần thuê"` hoặc đơn giá `< 15 triệu/m²` (bắt trọn các dòng nhập `10,500,000` thay vì `10,500,000,000` hoặc nhập nhầm đơn giá/m² vào ô tổng giá).
            - **Ngưỡng `Căn hộ chung cư`**: Diện tích `[18, 500] m²`, tổng giá `[0.8 tỷ, 100 tỷ]`, đơn giá `[18, 350] triệu/m²`.
            - **Ngưỡng `Nhà`**: Diện tích `[12, 1000] m²`, tổng giá `[0.8 tỷ, 600 tỷ]`, đơn giá `[15, 2500] triệu/m²` (bắt trọn lỗi nhân đôi `tổng giá × diện tích` như căn nhà 62m² giá 9.35 tỷ bị nhập thành 579.7 tỷ, hoặc nhập nhầm mặt tiền `3.7m` vào diện tích `3.7m²`, trong khi vẫn giữ lại các khách sạn/nhà mặt phố cổ Hoàn Kiếm hợp lệ `1,500 – 2,400 triệu/m²`!).
    - **Chốt chặn chống Target Leakage**: Biến `ppm2` chỉ được tính cục bộ để kiểm tra điều kiện lọc và ghi vào báo cáo dòng bị loại `outliers_df`, tuyệt đối **không được gắn vào `kept_df`**.

5. **`sanitize_physical_attributes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]`**:
    - **Triết lý ML**: Khi một tin rao có `price`, `area`, `district_name` hoàn toàn hợp lệ nhưng người bán gõ nhầm `frontage_width = 45` (quên dấu chấm cho `4.5m` trên mảnh đất `40m²`) hoặc `bedroom_count = 96`, xóa cả dòng dữ liệu sẽ gây lãng phí mẫu. Thay vào đó, ta chuyển riêng ô sai vật lý về `np.nan` và cập nhật cờ `<col>_missing = 1`.
    - **Kiểm tra vật lý chiều sâu hàm ẩn**: Với `Nhà`, `implied_depth = area / frontage_width` bắt buộc `>= 1.5m` (không có ngôi nhà ở nào tại Hà Nội nông dưới 1.5m). Quy tắc này phát hiện chính xác **91 trường hợp** nhập sai mặt tiền (chủ yếu do quên dấu thập phân!).

6. **`deduplicate_reposts_and_assign_group_id(df: pd.DataFrame)`**:
    - **Khử Near-Duplicate Repost**:
        - `clean_text_for_dedup()` xóa token ẩn số điện thoại `[phone_number]`, dãy số điện thoại 9–11 số và toàn bộ dấu câu trước khi so khớp.
        - Sắp xếp theo thời gian đăng tăng dần (`keep="first"`), loại bỏ tin đăng lại nếu thuộc 1 trong 2 trường hợp:
            - **Luật 1 (`repost_same_title_key`)**: Trùng `(property_type_name, district_name, ward_name, round(area, 1), round(price/10tr), _name_clean)` với tiêu đề `>= 15` ký tự (`8,085` dòng repost bị loại).
            - **Luật 2 (`repost_same_description_key`)**: Trùng `(property_type_name, district_name, round(area, 1), _desc_clean)` với mô tả `>= 50` ký tự ngay cả khi môi giới sửa nhẹ tiêu đề hoặc điều chỉnh nhẹ giá chào (`3,891` dòng repost bị loại).
    - **Tạo khóa nhóm tài sản `property_group_id` (Không phụ thuộc giá)**:
        - Hàm `strip_price_tokens_for_group_key()` xóa mọi cụm giá tiền (`"12.5 tỷ"`, `"65tr/m2"`,...) khỏi tiêu đề rồi băm SHA-1 tổ hợp `(property_type_name | district_name | ward_name | round(area, 1) | title_sig_40)` thành chuỗi hex 16 ký tự `property_group_id`. Nhờ đó, nếu cùng một căn nhà được đăng lại sau 3 tuần với mức giá giảm nhẹ và mô tả viết lại một phần, chúng vẫn mang cùng `property_group_id` để Giai đoạn 4 ngăn chặn tuyệt đối việc lọt sang tập `Test`!

#### 3.2. File [clean_outliers.py](file:///d:/PROPERTY-VALUATION/clean_outliers.py)

Script điều phối thực thi Giai đoạn 3 từ đầu đến cuối: đọc `data/silver/hanoi_deduplicated.parquet`, gọi `run_cleaning_pipeline()`, ghi `data/silver/hanoi_cleaned.parquet`, xuất danh sách `12,420` bản ghi bị loại kèm lý do ra `reports/outlier_and_repost_report.csv` và lưu thống kê kiểm toán vào `reports/cleaning_summary.json`.

#### 3.3. File Kiểm thử [tests/test_raw_immutability.py](file:///d:/PROPERTY-VALUATION/tests/test_raw_immutability.py) & [tests/test_cleaning.py](file:///d:/PROPERTY-VALUATION/tests/test_cleaning.py)

- `test_raw_files_exist_and_preserve_exact_row_counts`: Đọc metadata Parquet xác nhận `data/raw/shard_0000.parquet` giữ nguyên `350,075 × 19` và `data/raw/hanoi_housing_raw.parquet` giữ nguyên `93,940 × 19`.
- 5 unit test trong `test_cleaning.py`: Kiểm chứng gộp hướng nhà có gạch nối, giữ tên đường đánh số (`Đường 800A`, `Đường 70`), lọc lỗi đơn vị 1000x & không rò rỉ cột `ppm2`, sanitize mặt tiền phi vật lý mà không mất dòng, và khử chính xác cặp tin repost lệch nhau chuỗi `[phone_number]` (giữ tin đăng sớm hơn).

---

### 4. Cách Chạy & Kết Quả Kiểm Thử Thực Tế Giai Đoạn 3

- **Lệnh chạy pipeline làm sạch**:
    ```powershell
    .\.venv\Scripts\python.exe clean_outliers.py
    ```
- **Lệnh chạy kiểm thử tự động**:
    ```powershell
    .\.venv\Scripts\python.exe -m pytest -v tests/test_raw_immutability.py tests/test_cleaning.py
    ```
- **Kết quả thực tế (trích xuất từ `reports/cleaning_summary.json` & `pytest`)**:
    - Đầu vào: **`90,073` dòng** (`data/silver/hanoi_deduplicated.parquet`).
    - Loại bỏ ngoại lai & sai đơn vị / tin thuê: **`444` dòng** (`unit_error_or_rental_ppm2_lt_15m`: `264`, `house_area_out_of_bounds`: `57`, `house_price_out_of_bounds`: `37`, `apartment_ppm2_out_of_bounds`: `28`, `apartment_area_out_of_bounds`: `20`, `apartment_price_out_of_bounds`: `17`, `rental_title_misclassified`: `12`, `house_ppm2_out_of_bounds`: `9`).
    - Sanitize ô vật lý bất hợp lý về `NaN` (không mất dòng): `frontage_width`: `91`, `bedroom_count`: `35`, `bathroom_count`: `29`, `road_width`: `13`, `house_depth`: `3`.
    - Loại bỏ tin môi giới đăng lại (Near-Repost): **`11,976` dòng** (`repost_same_title_key`: `8,085`, `repost_same_description_key`: `3,891`).
    - Đầu ra sạch: **`77,653` dòng × `26` cột** tại `data/silver/hanoi_cleaned.parquet` (`Nhà`: **`55,259`**, `Căn hộ chung cư`: **`22,394`**, gồm **`76,493` `property_group_id`** độc lập).
    - Kết quả `pytest`: **`6 passed in 1.57s` (100% PASS)**.

---

## PHẦN III: GIAI ĐOẠN 4 (P0) — CHIA TẬP PURGED CHRONOLOGICAL SPLIT (`GOLD LAYER`) & PIPELINE TRÍCH XUẤT ĐẶC TRƯNG CHỐNG DATA LEAKAGE

### 1. Tư duy Thiết kế Chống Data Leakage Toàn Diện (3 Lớp Phòng Thủ)

Trong bài toán Định giá Bất động sản thực tế, có **3 nguồn rò rỉ dữ liệu tinh vi nhất** thường khiến mô hình trong phòng thí nghiệm đạt điểm giả tạo rất cao nhưng thất bại khi đưa vào vận hành (Production):

1. **Temporal & Cross-Period Repost Leakage (Rò rỉ thời gian & đăng lại xuyên kỳ)**:
    - Nếu chia ngẫu nhiên (`train_test_split` thông thường), mô hình dùng giao dịch tương lai (tháng 7/2025) để đoán quá khứ (tháng 6/2025).
    - Thậm chí ngay cả khi chia theo mốc thời gian thuần túy, nếu một căn nhà đã đăng ngày `15/06/2025` (nằm ở `Train`) được môi giới đăng lại ngày `20/07/2025` với giá giảm nhẹ (nằm ở `Test`), mô hình đã nhìn thấy chính căn nhà đó trong `Train`!
    - **Giải pháp (`Purged Chronological Split`)**: Áp dụng kỹ thuật **Purged Walk-Forward** (chuẩn mực trong Tài chính định lượng — Marcos López de Prado): cắt nghiêm ngặt theo trục thời gian `70% / 15% / 15%`, sau đó **thanh lọc (purge)** khỏi `Val` mọi bản ghi có `property_group_id` đã xuất hiện trong `Train`, và thanh lọc khỏi `Test` mọi bản ghi có `property_group_id` đã xuất hiện trong `Train ∪ Val`. Kết quả đạt đồng thời 2 bất biến toán học:
      $$\max(t_{\text{train}}) < \min(t_{\text{val}}) < \max(t_{\text{val}}) < \min(t_{\text{test}}) \quad \text{và} \quad G_{\text{train}} \cap G_{\text{val}} = G_{\text{train}} \cap G_{\text{test}} = G_{\text{val}} \cap G_{\text{test}} = \varnothing$$

2. **Text Target Leakage (Rò rỉ giá bán từ tiêu đề & mô tả tin rao)**:
    - Hơn 80% tiêu đề tin đăng chứa thẳng đáp án (VD: *"Bán nhà Trung Kính 52m2*6T giá 14.5 tỷ"\*). Trong khi đó, 77% bản ghi khuyết `floor_count` và 57% khuyết `frontage_width` ở cột cấu trúc nhưng lại ghi rõ `"6T"`, `"MT 4.5m"`, `"thang máy"`, `"ô tô tránh"` trong câu chữ!
    - **Giải pháp (`Price-Redacted Domain NLP`)**: Trước khi chạy bất kỳ biểu thức chính quy trích xuất đặc trưng nào, hàm `redact_price_mentions()` xóa sạch mọi biểu thức tiền tệ tiếng Việt (`14.5 tỷ`, `8ty5`, `280tr/m2`, `4,9x tỷ`, `700 triệu`, `giá chào...`), và tuyệt đối không dùng độ dài ký tự thô. Bài test toán học chứng minh: thay `"giá 14.5 tỷ"` thành `"giá 999 tỷ"` hay xóa hẳn giá đều cho ra vector đặc trưng **giống hệt nhau từng bit (`100%` bit-for-bit identical)**!

3. **High-Cardinality Spatial Target Leakage (Rò rỉ khi mã hóa 1,072 Đường phố & 837 Dự án)**:
    - Nếu tính đơn giá trung bình theo `street_name` hoặc `project_name` trên toàn tập (hoặc tính trực tiếp trên `Train` mà không che dòng hiện tại), các con phố/dự án ít mẫu (1–3 tin) sẽ sao chép trực tiếp nhãn `price` vào đặc trưng đầu vào, gây Overfitting cực nặng!
    - **Giải pháp (`5-Fold Out-Of-Fold Hierarchical Empirical Bayes Target Encoding`)**:
        - Chỉ học trên `X_train` và biến mục tiêu chuẩn hóa quy mô $y_{\text{unit}} = \log(\text{price} / \text{area})$ (log đơn giá/m² giúp tách biệt giá trị vị trí đất khỏi diện tích căn nhà!).
        - Trên `Train`, chia 5 Fold: đặc trưng của mẫu ở Fold $k$ **chỉ được tính từ 4 Fold còn lại**.
        - Làm trơn phân cấp theo công thức **Empirical Bayes (Shrinkage)**:
          $$\hat{\mu}_{\text{con}} = \frac{n \cdot \bar{y}_{\text{con}} + m \cdot \hat{\mu}_{\text{cha}}}{n + m}$$
          theo chuỗi phân cấp 5 tầng: `Loại BĐS` $\rightarrow$ `(Loại BĐS, Quận)` ($m=20$) $\rightarrow$ `(Loại BĐS, Quận, Phường)` ($m=15$) $\rightarrow$ `(Loại BĐS, Quận, Đường)` ($m=15$) & `(Loại BĐS, Quận, Dự án)` ($m=10$). Khi suy diễn trên `Val`, `Test` hoặc căn nhà mới ngoài thực tế, pipeline chỉ tra bảng thống kê đã đóng băng từ `Train` và tự động lùi về cấp cha nếu gặp tên đường/dự án mới chưa từng xuất hiện!

---

### 2. Luồng Dữ liệu Giai đoạn 4

```
data/silver/hanoi_cleaned.parquet (77,653 dòng × 26 cột)
   │
   ▼  [src/splitting.py :: purged_chronological_split()]
   ├──► data/gold/train.parquet (54,357 dòng: 01/06/2025 → 28/06/2025)
   ├──► data/gold/val.parquet   (11,480 dòng: 28/06/2025 → 14/07/2025 | đã purge 168 repost xuyên kỳ)
   └──► data/gold/test.parquet  (11,498 dòng: 14/07/2025 → 30/07/2025 | đã purge 150 repost xuyên kỳ)
          │
          ▼  [src/features.py :: HanoiValuationFeaturePipeline]
          • .fit_transform(train_df, y_train)  ──► X_train (54,357 × 110 đặc trưng, OOF Spatial + Medians Train)
          • .transform(val_df)                 ──► X_val   (11,480 × 110 đặc trưng, tra bảng đóng băng từ Train)
          • .transform(test_df)                ──► X_test  (11,498 × 110 đặc trưng, tra bảng đóng băng từ Train)
```

---

### 3. Giải thích Chi tiết Từng File, Lớp & Hàm trong Giai đoạn 4

#### 3.1. File [src/splitting.py](file:///d:/PROPERTY-VALUATION/src/splitting.py)

- **Hàm `purged_chronological_split(df, train_frac=0.70, val_frac=0.15)`**:
    - Sắp xếp ổn định (`kind="stable"`) theo `(_dt, property_group_id)`.
    - Xác định mốc thời gian cắt `t_train_end` (phân vị 70%) và `t_val_end` (phân vị 85%), chia bằng bất đẳng thức ngặt `>` để đảm bảo không trùng lặp dù chỉ 1 microsecond tại biên cắt.
    - Thực hiện lọc tập hợp `isin(train_groups)` trên `raw_val` và `isin(train_val_groups)` trên `raw_test`.
    - Chứa 5 câu lệnh `assert` toán học kiểm tra trực tiếp tính thứ tự thời gian và tính rời nhau đôi một (`isdisjoint`) của `property_group_id` trước khi trả về kết quả.

#### 3.2. File [src/features.py](file:///d:/PROPERTY-VALUATION/src/features.py)

1. **`redact_price_mentions(series: pd.Series) -> pd.Series`**:
    - Xóa mọi mẫu giá tiền (dạng rút gọn `8ty5`, dạng đầy đủ `12.5 tỷ`, `67tr/m2`, `4,9x tỷ`, `700 triệu`, và các cụm từ khóa `giá bán`, `giá nhỉnh`,...) trước khi trích xuất NLP.
2. **`extract_stateless_features(df: pd.DataFrame) -> pd.DataFrame`**:
    - **Khôi phục thông số vật lý bị khuyết từ văn bản đã che giá**:
        - Số tầng (`ext_floor`): bắt cả mẫu `"5 tầng"`, `"6 lầu"` lẫn cú pháp viết tắt đặc thù của môi giới Hà Nội `"52m2*6T"`, `"45m2x5t"`! Kiểm tra chặn vật lý `[1, 25]` cho `Nhà` và `[1, 45]` cho `Căn hộ chung cư`. Kết hợp với cột gốc qua `fc_orig.fillna(ext_floor)` và bật cờ `floor_from_text`.
        - Tương tự khôi phục mặt tiền (`ext_frontage` kèm điều kiện chiều sâu hàm ẩn `area / frontage >= 1.5m`), độ rộng đường/ngõ (`ext_road`), số phòng ngủ (`ext_bed`), số phòng tắm (`ext_bath`).
    - **Đặc trưng hình học & tỷ lệ kiến trúc**:
        - `log_area`, `sqrt_area`: Nắm bắt độ co giãn phi tuyến của giá theo diện tích.
        - `total_floor_area_est` & `log_total_floor_area_est`: Ước tính tổng diện tích sàn xây dựng (`area * floor_count_eff` cho `Nhà`, `area` cho `Căn hộ chung cư`).
        - `implied_depth = area / frontage_width_eff` & `shape_aspect_ratio = frontage_width_eff / implied_depth`: Đo độ vuông vắn của thửa đất (nhà mặt tiền rộng nông đắt hơn nhà ống hẹp sâu!).
        - `area_per_bedroom`, `bath_per_bedroom`: Đo phân khúc căn hộ cao cấp (phòng rộng) so với căn hộ chia nhỏ phòng.
    - **15 Tín hiệu Vị trí / Hạ tầng / Pháp lý nhị phân & Thang bậc Tiếp cận `access_tier` (`0..4`)**:
        - Trích xuất 15 biến `0/1`: `is_mat_pho`, `is_phan_lo`, `is_oto_tranh`, `is_oto_vao_nha`, `is_ngo_thong`, `is_lo_goc`, `has_thang_may`, `is_kinh_doanh`, `is_chdv_ccmn`, `has_so_do`, `has_full_noi_that`, `is_nha_moi`, `is_view_dep`, `is_biet_thu_lk`, `is_studio_penthouse`.
        - Tổng hợp thành biến thứ bậc `access_tier` (`0`: ngõ nhỏ thường $\rightarrow$ `1`: ngõ thông/ba gác $\rightarrow$ `2`: ô tô vào nhà/gara $\rightarrow$ `3`: ô tô tránh/vỉa hè $\rightarrow$ `4`: mặt phố lớn) và biến tương tác `house_x_access_tier` (vì độ rộng ngõ quyết định giá `Nhà` thổ cư chứ không quyết định giá `Căn hộ chung cư`!).
3. **Lớp `HierarchicalOOFTargetEncoder(BaseEstimator, TransformerMixin)`**:
    - Cài đặt đầy đủ giao diện `scikit-learn` (`fit`, `fit_transform`, `transform`).
    - Sinh **15 đặc trưng không gian địa lý**: log đơn giá làm trơn ở 5 cấp (`type`, `district`, `ward`, `street`, `project`), độ lệch chuẩn nội quận `te_district_std`, tần suất xuất hiện logarit trong tập Train (`*_log_train_count`), chênh lệch vị trí vi mô (`ward_vs_district_diff`, `street_vs_ward_diff`, `project_vs_ward_diff`), cùng mỏ neo không gian kết hợp `spatial_anchor_log_ppm2` và `spatial_anchor_log_tot = spatial_anchor_log_ppm2 + log_area`!
4. **Lớp `HanoiValuationFeaturePipeline(BaseEstimator, TransformerMixin)`**:
    - Ghép nối toàn bộ quy trình: `_sanitize_input` (tự động loại bỏ `price` và `property_group_id` ngay cửa vào) $\rightarrow$ `extract_stateless_features` $\rightarrow$ `HierarchicalOOFTargetEncoder` $\rightarrow$ One-Hot Encoding cho `district_name`, `house_direction`, `balcony_direction` (chỉ học danh mục từ `Train`) $\rightarrow$ `_verify_no_forbidden_columns` (chốt chặn `AssertionError` nếu có bất kỳ cột nào chứa từ `"price"`).
    - Cung cấp phương thức `impute_with_train_medians()` điền `NaN` bằng trung vị riêng của `Nhà` và `Căn hộ chung cư` học **độc lập trên `X_train`** để phục vụ các mô hình tuyến tính (`Ridge`).
    - _Chủ động sửa cảnh báo tương thích tương lai_: Đã cập nhật `.dt.dayofweek` sang thuộc tính chuẩn mới `.dt.day_of_week` để tương thích hoàn toàn với lộ trình Pandas 3/4 (0 warning khi chạy `pytest`).

#### 3.3. File [split_dataset.py](file:///d:/PROPERTY-VALUATION/split_dataset.py) & [tests/test_no_leakage.py](file:///d:/PROPERTY-VALUATION/tests/test_no_leakage.py)

- `split_dataset.py`: Thực thi chia tập lưu vào `data/gold/{train,val,test}.parquet`, chạy kiểm tra pipeline đặc trưng và lưu thống kê vào `reports/split_summary.json`.
- `tests/test_no_leakage.py`: Bộ 3 kiểm thử chống rò rỉ dữ liệu nghiêm ngặt:
    1. `test_gold_splits_strict_chronology_and_zero_group_overlap`: Xác minh thứ tự thời gian tuyệt đối và giao các tập `property_group_id` bằng rỗng.
    2. `test_nlp_feature_extraction_is_invariant_to_asking_price_mentions`: Đổi giá trong tiêu đề/mô tả từ `14.5 tỷ (280tr/m2)` sang `999 tỷ` và xóa trắng giá, chứng minh 100% đặc trưng trích xuất không đổi.
    3. `test_oof_target_encoder_does_not_leak_own_row_or_test_set`: Tăng vọt giá của dòng số 0 gấp 40 lần (`500 tỷ`) trên một con phố duy nhất và chứng minh giá trị mã hóa OOF của chính dòng số 0 không hề thay đổi!

---

### 4. Cách Chạy & Kết Quả Kiểm Thử Thực Tế Giai Đoạn 4

- **Lệnh chạy chia tập & kiểm tra đặc trưng**:
    ```powershell
    .\.venv\Scripts\python.exe split_dataset.py
    ```
- **Lệnh chạy kiểm thử chống Data Leakage**:
    ```powershell
    .\.venv\Scripts\python.exe -m pytest -v tests/test_no_leakage.py
    ```
- **Kết quả thực tế (từ `reports/split_summary.json` & `pytest`)**:
    - **Tập `Train`**: **`54,357` dòng** (`Nhà`: `39,066`, `Căn hộ chung cư`: `15,291`) | Thời gian: `2025-06-01T00:09:45Z` $\rightarrow$ `2025-06-28T07:41:33.938Z`.
    - **Tập `Validation`**: **`11,480` dòng** (`Nhà`: `8,169`, `Căn hộ chung cư`: `3,311`, đã thanh lọc **`168` repost xuyên kỳ**) | Thời gian: `2025-06-28T07:41:54.479Z` $\rightarrow$ `2025-07-14T10:24:16.981Z`.
    - **Tập `Test`**: **`11,498` dòng** (`Nhà`: `7,860`, `Căn hộ chung cư`: `3,638`, đã thanh lọc **`150` repost xuyên kỳ**) | Thời gian: `2025-07-14T10:24:38.140Z` $\rightarrow$ `2025-07-30T23:51:12.138Z`.
    - **Hiệu quả khôi phục dữ liệu vật lý từ văn bản riêng trên `Train`**: Khôi phục thành công **`27,056` giá trị `floor_count`**, **`4,832` giá trị `bedroom_count`**, **`2,466` giá trị `road_width`**, và **`1,998` giá trị `frontage_width`** bị bỏ trống!
    - **Tổng số chiều đặc trưng đầu ra**: **`110` đặc trưng số sạch (0% cột mục tiêu)**.
    - **Kết quả `pytest`**: **`9/9 passed` (bao gồm 3 bài test toán học chống rò rỉ dữ liệu)**.

---

## PHẦN IV: GIAI ĐOẠN 5 (P0) — HUẤN LUYỆN HỆ THỐNG MÔ HÌNH ĐA TẦNG, CHỌN MÔ HÌNH TRÊN VALIDATION & ĐÁNH GIÁ BOOTSTRAP CI TRÊN TEST

### 1. Kiến thức Toán học & ML Cốt lõi trong Định giá Bất động sản (`AVM Modeling`)

1. **Tại sao dự đoán Log Đơn Giá $\log(\text{price} / \text{area})$ thay vì dự đoán trực tiếp `price` (VNĐ)?**
    - **Phân phối lệch phải cực đoan (Heavy Right-Skewness)**: Giá BĐS Hà Nội trải dài từ `0.8 tỷ` đến `500 tỷ` (chênh nhau hơn 600 lần!). Nếu tối ưu MSE trực tiếp trên `price`, sai số `15%` ở một khách sạn `200 tỷ` (`30 tỷ` sai số bình phương $= 900$) lớn gấp **40,000 lần** sai số `15%` ở một căn hộ `1 tỷ` (`0.15 tỷ` sai số bình phương $= 0.0225$). Mô hình sẽ hy sinh toàn bộ phân khúc nhà dân sinh để chiều lòng vài chục căn biệt thự/khách sạn!
    - **Khắc phục nhược điểm hằng số từng khúc của Cây quyết định (Piecewise-Constant Extrapolation)**: Cây quyết định (Decision Tree / GBDT) chia không gian thành các hình hộp chữ nhật. Nếu dự đoán trực tiếp $\log(\text{price})$, cây phải tốn hàng trăm nút cắt chỉ để xấp xỉ hàm tuyến tính $+\log(\text{area})$ và bị bão hòa trần ở các mảnh đất diện tích lớn. Khi đặt biến mục tiêu huấn luyện là:
      $$y_{\text{unit}} = \log\left(\frac{\text{price}}{\text{area}}\right) \implies \hat{P} = \alpha \cdot \text{area} \cdot \exp(\hat{y}_{\text{unit}})$$
      mọi nút cắt của LightGBM tập trung 100% dung lượng vào **chất lượng vi vị trí, độ rộng ngõ/mặt phố, số tầng, mặt tiền, thang máy và hiệu ứng quy mô tinh tế**, đồng thời ngoại suy mượt mà tỷ lệ thuận với `area` (`area` là đầu vào đã biết 100% lúc định giá, không gây rò rỉ nhãn!).
    - **Hàm mất mát Huber trên thang Logarit (`objective="huber"`)**:
        - Sai số trên thang log $|\log \hat{P} - \log P| = |\log(\hat{P}/P)| \approx |\hat{P} - P| / P$ xấp xỉ trực tiếp **sai số phần trăm tương đối (MAPE / MdAPE)**!
        - Hàm **Huber Loss** kết hợp ưu điểm khả vi bậc hai mượt mà quanh điểm 0 của MSE ($L_2$) với độ bền vững tuyến tính trước nhiễu ngoại lai ở hai đuôi của MAE ($L_1$).
    - **Hiệu chỉnh hệ số nhân $\alpha$ theo Bất đẳng thức Jensen**: Do $\mathbb{E}[\exp(Y)] \neq \exp(\mathbb{E}[Y])$, hàm `tune_multiplicative_mape_factor()` tìm hệ số hiệu chỉnh $\alpha^* \in [0.90, 1.05]$ **chỉ trên tập `Validation`** để tối ưu cân bằng `0.5 * MAPE + 0.5 * MdAPE`.

2. **Tại sao kiến trúc Chuyên gia Phân tách (`Segmented LightGBM`: `HouseExpert` + `ApartmentExpert`) vượt trội hơn Mô hình Gộp đơn lẻ?**
    - Trong thực tế Hà Nội, `Căn hộ chung cư` được định giá bởi `project_name`, `ward_name`, số phòng ngủ, hướng ban công, tầm nhìn (`is_view_dep`), nội thất; trong khi các biến sống còn của `Nhà` thổ cư như `is_mat_pho`, `is_oto_tranh`, `frontage_width_eff`, `floor_count_eff`, `shape_aspect_ratio` lại hoàn toàn vô nghĩa với chung cư!
    - Tách 2 mô hình LightGBM chuyên biệt (`SegmentedLightGBMValuationModel`) rồi phối hợp trung bình nhân hình học trên không gian log với mô hình toàn cục (`UnifiedLightGBMValuationModel`):
      $$\hat{P}_{\text{ens}} = \exp\Big(w_{\text{seg}} \log \hat{P}_{\text{seg}} + (1 - w_{\text{seg}}) \log \hat{P}_{\text{uni}}\Big)$$
      với trọng số $w_{\text{seg}} = 0.70$ được chọn tự động **trên `Validation`** giúp vừa khai thác cấu trúc chuyên biệt từng phân khúc vừa tận dụng tri thức không gian chung giữa 30 quận huyện!

---

### 2. Giải thích Chi tiết Từng File, Lớp & Hàm trong Giai đoạn 5

#### 2.1. File [src/modeling.py](file:///d:/PROPERTY-VALUATION/src/modeling.py)

1. **`compute_avm_metrics(y_true_vnd, y_pred_vnd)`**:
    - Tính bộ **11 chỉ số chuẩn công nghiệp AVM quốc tế**:
        - `MAPE_pct` (Sai số phần trăm tuyệt đối trung bình) & `MdAPE_pct` (Sai số phần trăm tuyệt đối trung vị — chỉ số tiêu chuẩn số 1 của Zillow Zestimate).
        - `PPE10_pct`, `PPE15_pct`, `PPE20_pct` (_Percentage of Predictions within $\pm 10\%, \pm 15\%, \pm 20\%$_ — tỷ lệ định giá trúng khung chấp nhận của thị trường).
        - `MAE_ty`, `MedAE_ty`, `RMSE_ty` (Sai số quy đổi ra đơn vị **Tỷ VNĐ**).
        - `R2_price` ($R^2$ trên giá tiền thực VNĐ), `R2_log` ($R^2$ trên thang logarit), `RMSLE`.
2. **`bootstrap_avm_confidence_intervals(y_true_vnd, y_pred_vnd, n_boot=1000, seed=42)`**:
    - Lấy mẫu lại có hoàn lại (_non-parametric bootstrap resampling_) `1,000` lần trên tập `Test` (`n = 11,498`), trích phân vị `[2.5%, 97.5%]` để xác lập khoảng tin cậy thống kê 95% không phụ thuộc giả định phân phối chuẩn.
3. **5 Tầng Mô hình + Tổ hợp Vô địch (`train_and_evaluate_all_models`)**:
    - `NaiveDistrictMedianBaseline`: Học đơn giá trung vị `(price / area)` theo từng cặp `(property_type_name, district_name)` trên `Train` rồi nhân với `area`. Đại diện cho cách ước lượng nhẩm truyền thống của người mua nhà.
    - `RidgeLogLinear`: `impute_with_train_medians` $\rightarrow$ `StandardScaler` $\rightarrow$ `Ridge(alpha=25.0)` trên 110 đặc trưng.
    - `SklearnHistGBR`: `HistGradientBoostingRegressor(loss="absolute_error")` xử lý trực tiếp `NaN`.
    - `UnifiedLightGBMValuationModel`: LightGBM toàn cục (`num_leaves=95`, `learning_rate=0.03`, Huber loss, `early_stopping=80` trên `Validation`, dừng tối ưu tại vòng `1,944`).
    - `SegmentedLightGBMValuationModel`: Hai chuyên gia LightGBM riêng biệt cho `Căn hộ chung cư` (`num_leaves=63`, dừng tối ưu vòng `2,064`) và `Nhà` (`num_leaves=95`, dừng tối ưu vòng `1,133`).
    - `BlendedGBDTValuationBundle`: Đóng gói đồng bộ cả `HanoiValuationFeaturePipeline` lẫn 2 mô hình LightGBM với trọng số $w_{\text{seg}} = 0.70$ thành một đối tượng duy nhất lưu xuống `models/champion_point_bundle.joblib` (loại bỏ hoàn toàn lệch pha Train–Serve!).
    - _Chủ động sửa cảnh báo_: Đã cập nhật API LightGBM 4.7 từ `eval_set=[(X_va, y_va)]` sang `eval_X=X_va, eval_y=y_va` để mã nguồn sạch tuyệt đối không còn `LGBMDeprecationWarning`.

#### 2.2. File [train_models.py](file:///d:/PROPERTY-VALUATION/train_models.py)

Điều phối đọc `data/gold/{train,val,test}.parquet`, huấn luyện toàn bộ 6 phương án, lưu mô hình vô địch ra `models/champion_point_bundle.joblib`, xuất bảng đối sánh `reports/model_comparison.csv` và báo cáo thống kê `reports/model_training_summary.json`.

---

### 3. Cách Chạy & Bảng Kết Quả Thực Nghiệm Giai Đoạn 5 (100% Dữ Liệu Thật)

- **Lệnh chạy huấn luyện & đánh giá**:
    ```powershell
    .\.venv\Scripts\python.exe train_models.py
    ```

#### Bảng 1: Kết quả trên tập `Validation` (`11,480` BĐS — Dùng để chọn mô hình & Early Stopping)

| Mô hình                            | Thời gian Fit (s) | MAPE (%) $\downarrow$ | MdAPE (%) $\downarrow$ | PPE10 (%) $\uparrow$ | PPE15 (%) $\uparrow$ | PPE20 (%) $\uparrow$ | MAE (Tỷ) $\downarrow$ | MedAE (Tỷ) $\downarrow$ | RMSE (Tỷ) $\downarrow$ | $R^2_{\text{price}}$ $\uparrow$ | $R^2_{\log}$ $\uparrow$ | RMSLE $\downarrow$ |
| :--------------------------------- | :---------------: | :-------------------: | :--------------------: | :------------------: | :------------------: | :------------------: | :-------------------: | :---------------------: | :--------------------: | :-----------------------------: | :---------------------: | :----------------: |
| `1_Naive_District_Median`          |       0.01        |         26.09         |         19.39          |        27.87         |        40.27         |        51.63         |         4.875         |          1.733          |         15.721         |              0.598              |          0.799          |       0.351        |
| `2_Ridge_LogLinear`                |       0.48        |         18.61         |         12.96          |        39.90         |        55.58         |        67.02         |         3.379         |          1.196          |         10.684         |              0.814              |          0.897          |       0.252        |
| `3_Sklearn_HistGBR`                |       6.03        |         15.76         |         10.42          |        48.54         |        63.50         |        74.56         |         2.953         |          0.958          |         9.564          |              0.851              |          0.918          |       0.224        |
| `4_LightGBM_Unified`               |       11.17       |         15.29         |         10.38          |        48.51         |        64.33         |        75.44         |         2.876         |          0.941          |         9.238          |              0.861              |          0.924          |       0.216        |
| `5_LightGBM_Segmented`             |       13.19       |         15.15         |         10.12          |        49.51         |        64.72         |        75.21         |         2.872         |          0.929          |         9.225          |              0.861              |          0.924          |       0.216        |
| **`6_Champion_LightGBM_Ensemble`** |     **24.36**     |       **15.11**       |       **9.997**        |      **50.02**       |      **64.94**       |      **75.49**       |       **2.863**       |        **0.931**        |       **9.214**        |            **0.862**            |        **0.925**        |     **0.215**      |

#### Bảng 2: Kết quả Đánh giá Độc lập trên tập `Test` Tương lai (`11,498` BĐS nửa cuối tháng 7/2025 — Untouched Holdout)

| Mô hình                            | MAPE (%) $\downarrow$ | MdAPE (%) $\downarrow$ | PPE10 (%) $\uparrow$ | PPE15 (%) $\uparrow$ | PPE20 (%) $\uparrow$ | MAE (Tỷ) $\downarrow$ | MedAE (Tỷ) $\downarrow$ | RMSE (Tỷ) $\downarrow$ | $R^2_{\text{price}}$ $\uparrow$ | $R^2_{\log}$ $\uparrow$ | RMSLE $\downarrow$ |
| :--------------------------------- | :-------------------: | :--------------------: | :------------------: | :------------------: | :------------------: | :-------------------: | :---------------------: | :--------------------: | :-----------------------------: | :---------------------: | :----------------: |
| `1_Naive_District_Median`          |         25.49         |         19.08          |        28.75         |        41.33         |        52.11         |         4.400         |          1.656          |         14.101         |              0.612              |          0.796          |       0.339        |
| `2_Ridge_LogLinear`                |         18.11         |         12.90          |        40.22         |        56.10         |        68.01         |         3.103         |          1.135          |         10.588         |              0.781              |          0.894          |       0.244        |
| `3_Sklearn_HistGBR`                |         15.78         |         10.80          |        47.10         |        63.32         |        74.49         |         2.775         |          0.937          |         9.332          |              0.830              |          0.913          |       0.221        |
| `4_LightGBM_Unified`               |         15.21         |         10.47          |        48.03         |        64.34         |        75.04         |         2.660         |          0.898          |         8.860          |              0.847              |          0.920          |       0.212        |
| `5_LightGBM_Segmented`             |         14.96         |         10.22          |        49.13         |        65.24         |        75.71         |         2.656         |          0.895          |         8.923          |              0.845              |          0.921          |       0.210        |
| **`6_Champion_LightGBM_Ensemble`** |       **14.95**       |       **10.21**        |      **49.04**       |      **65.22**       |      **75.68**       |       **2.649**       |        **0.887**        |       **8.891**        |            **0.846**            |        **0.921**        |     **0.210**      |

#### Bảng 3: Khoảng Tin Cậy 95% Bootstrap (`1,000` lần lấy mẫu) trên Tập `Test`

| Chỉ số            |   Naive Baseline (95% CI)   | Champion LightGBM Ensemble (95% CI) | Mức cải thiện tuyệt đối / tương đối                           |
| :---------------- | :-------------------------: | :---------------------------------: | :------------------------------------------------------------ |
| **MAPE (%)**      | `25.49%` `[24.88%, 26.05%]` |   **`14.95%` `[14.59%, 15.31%]`**   | **Giảm `-10.54` điểm %** (giảm **`41.3%`** sai số tương đối!) |
| **MdAPE (%)**     | `19.08%` `[18.56%, 19.46%]` |   **`10.21%` `[10.01%, 10.42%]`**   | **Giảm `-8.87` điểm %** (50% BĐS sai số dưới `10.21%`)        |
| **PPE15 (%)**     | `41.33%` `[40.52%, 42.33%]` |   **`65.22%` `[64.33%, 66.11%]`**   | **Tăng `+23.89` điểm %**                                      |
| **PPE20 (%)**     | `52.10%` `[51.29%, 53.05%]` |   **`75.68%` `[74.90%, 76.47%]`**   | **Tăng `+23.58` điểm %** (>3/4 BĐS nằm gọn trong $\pm 20\%$)  |
| **MAE (Tỷ VNĐ)**  |  `4.400` `[4.171, 4.635]`   |    **`2.649` `[2.502, 2.799]`**     | **Giảm `-1.752 tỷ VNĐ`** mỗi giao dịch                        |
| **RMSE (Tỷ VNĐ)** | `14.101` `[12.098, 16.071]` |    **`8.891` `[7.399, 10.273]`**    | **Giảm `-5.209 tỷ VNĐ`**                                      |
| **$R^2_{\log}$**  |  `0.796` `[0.788, 0.804]`   |    **`0.921` `[0.917, 0.926]`**     | **Tăng `+0.126`** (giải thích **92.15%** phương sai log giá!) |

> **Nhận xét kiểm chứng tính tổng quát hóa (Zero Overfitting)**: Các chỉ số trên tập `Validation` (`MAPE = 15.11%`, `MdAPE = 10.00%`, $R^2_{\log} = 0.925$) và tập `Test` tương lai (`MAPE = 14.95%`, `MdAPE = 10.21%`, $R^2_{\log} = 0.921$) bám sát nhau gần như tuyệt đối! Điều này chứng minh pipeline OOF Target Encoding + Purged Chronological Split không hề bị rò rỉ hay học vẹt, và mô hình duy trì độ chính xác ổn định theo thời gian.

---

## PHẦN V: GIAI ĐOẠN 6 (P1) — DỰ ĐOÁN KHOẢNG GIÁ MONDRIAN SPLIT CONFORMAL QUANTILE (`[P10, P90]`), PHÂN TÍCH LỖI THEO LÁT CẮT & ĐỘ QUAN TRỌNG ĐẶC TRƯNG

### 1. Lý thuyết Toán học: Tại sao dùng Mondrian Split Conformalized Quantile Regression (CQR)?

Trong thẩm định giá ngân hàng và tư vấn giao dịch thực tế, một con số điểm đơn lẻ (VD: `7.20 tỷ VNĐ`) không phản ánh được mức độ không chắc chắn (_epistemic & aleatoric uncertainty_) của từng loại tài sản: một căn hộ `55m²` tại _Vinhomes Smart City_ có mặt bằng giá cực kỳ đồng nhất (biên độ dao động hẹp `±11%`), trong khi một ngôi nhà phố cổ _Hoàn Kiếm_ hoặc biệt thự _Tây Hồ_ có biên độ thương lượng rộng hơn nhiều!

Nếu chỉ dùng **Quantile Regression** thuần túy (`LightGBM objective="quantile"` tại $\alpha = 0.10$ và $\alpha = 0.90$), cây quyết định thường gặp 2 nhược điểm kinh điển:

1. **Thiếu độ phủ ngoài mẫu (Undercoverage)**: Khoảng `[P10, P90]` lý thuyết 80% thường chỉ đạt 70%–75% khi sang tập `Test` tương lai.
2. **Hiện tượng cắt chéo phân vị (Quantile Crossing)**: Ở các vùng thưa dữ liệu, có thể xảy ra nghịch lý $\hat{q}_{0.10}(x) > \hat{y}_{\text{point}}(x)$.

**Giải pháp `MondrianConformalValuationSystem` (Romano, Patterson & Candès — NeurIPS 2019)**:

1. Huấn luyện 2 mô hình phân vị `q_lo_model_` ($\alpha = 0.10$, Pinball Loss) và `q_hi_model_` ($\alpha = 0.90$) **chỉ trên tập `Train`** (dừng sớm trên `Validation`).
2. Ép tính bao hàm đơn điệu quanh dự đoán điểm trên thang log đơn giá:
   $$\tilde{q}_{\text{lo}}(x) = \min\big(\hat{q}_{0.10}(x),\, \hat{y}_{\text{point}}(x)\big), \qquad \tilde{q}_{\text{hi}}(x) = \max\big(\hat{q}_{0.90}(x),\, \hat{y}_{\text{point}}(x)\big)$$
3. Trên tập hiệu chỉnh độc lập **`Validation`**, tính điểm bất tương thích (_non-conformity score_) cho từng mẫu $i$:
   $$E_i = \max\Big(\tilde{q}_{\text{lo}}(x_i) - y_i,\; y_i - \tilde{q}_{\text{hi}}(x_i)\Big)$$
4. **Phân tầng Mondrian theo loại hình BĐS**: Tính ngưỡng bù Conformal hữu hạn mẫu $\hat{Q}_{0.80}$ tại phân vị $\frac{\lceil (n_{\text{cal}} + 1) \cdot 0.80 \rceil}{n_{\text{cal}}}$ **tách riêng** cho `Căn hộ chung cư` ($\hat{Q}_{\text{apt}} = +0.0147$ log-unit) và `Nhà` ($\hat{Q}_{\text{house}} = +0.0334$ log-unit).
5. Khoảng định giá cuối cùng đảm bảo **100% không bao giờ cắt chéo** (`P10 < Fair Price < P90`) và giữ độ sắc nét riêng cho từng phân khúc!

---

### 2. Giải thích Chi tiết Từng File & Từng Hàm trong Giai đoạn 6

#### 2.1. File [src/intervals_and_analysis.py](file:///d:/PROPERTY-VALUATION/src/intervals_and_analysis.py)

- **`MondrianConformalValuationSystem`**:
    - `.fit_and_calibrate(train_df, val_df)`: Huấn luyện 2 mô hình phân vị trên `Train` và học $\hat{Q}_{\text{apt}}, \hat{Q}_{\text{house}}$ trên `Validation`.
    - `.predict_with_intervals(raw_df)`: Xuất bảng kết quả gồm `pred_low_vnd` (`P10`), `pred_price_vnd` (giá hợp lý trung tâm), `pred_high_vnd` (`P90`), `pred_ppm2_million` (đơn giá triệu/m²) và `interval_width_pct` (độ rộng tương đối `(P90 - P10) / Fair_Price * 100%`).
- **`compute_interval_diagnostics(y_true_vnd, pred_df)`**: Đo lường xác suất bao phủ thực nghiệm `PICP_80_pct`, tỷ lệ rơi dưới `P10`, vượt trên `P90`, độ rộng khoảng trung vị và tỷ lệ không cắt chéo `monotonic_non_crossing_pct`.
- **`compute_slice_error_analysis(test_df, pred_df, feat_df)`**: Phân rã sai số trên tập `Test` theo 4 chiều nghiệp vụ trọng yếu: (1) Loại hình BĐS, (2) 6 Phân khúc giá từ `< 3 tỷ` đến `> 60 tỷ`, (3) Mức độ đầy đủ thông tin cấu trúc (0–1 cột, 2–3 cột, 4–5 cột), và (4) Từng Quận/Huyện có từ 25 mẫu trở lên.
- **`extract_feature_importance_table(bundle)`**: Trích xuất tỷ trọng đóng góp hàm mục tiêu (`% Gain`) chuẩn hóa tổng bằng 100% trên cả 3 mô hình `Unified`, `House Expert`, và `Apartment Expert` để so sánh cơ chế định giá giữa hai loại hình.

#### 2.2. File [analyze_and_calibrate.py](file:///d:/PROPERTY-VALUATION/analyze_and_calibrate.py)

Chạy toàn bộ quy trình hiệu chỉnh CQR, lưu hệ thống hoàn chỉnh vào `models/avm_conformal_system.joblib` và xuất 3 báo cáo: `reports/slice_error_analysis.csv`, `reports/feature_importance.csv`, `reports/conformal_and_slice_summary.json`.

---

### 3. Phân tích Chuyên sâu Kết quả Thực tế Giai đoạn 6 (Câu chuyện Dữ liệu BĐS Hà Nội)

- **Lệnh chạy**:
    ```powershell
    .\.venv\Scripts\python.exe analyze_and_calibrate.py
    ```

#### 3.1. Chất lượng Khoảng Tin Cậy Định Giá 80% (Mondrian CQR)

- Trên `Validation`: `PICP_80_pct = 80.03%`, không cắt chéo `100.00%`.
- Trên tập `Test` tương lai (`11,498` BĐS chưa từng thấy):
    - **Độ phủ tổng thể `PICP_80_pct = 78.67%`** (riêng phân khúc `Nhà` đạt **`79.91%`** — khớp gần như hoàn hảo với mục tiêu lý thuyết `80.00%`!).
    - **Tỷ lệ không cắt chéo (`P10 < Point < P90`)**: **`100.00%` (`11,498 / 11,498` tài sản)**.
    - **Độ sắc nét thích nghi**: Khoảng giá 80% của `Căn hộ chung cư` có độ rộng trung vị chỉ **`24.60%`** (xấp xỉ `±12.3%` quanh giá định giá), trong khi `Nhà` thổ cư tự động mở rộng lên **`53.93%`** để phản ánh đúng tính dị biệt về hình dáng thửa đất, nội thất và độ sâu ngõ ngách!

#### 3.2. Phân tích Lỗi Theo Lát Cắt (Slice-Based Error Analysis trên Tập `Test`)

1. **Theo Loại hình Bất động sản**:
    - **`Căn hộ chung cư` (`3,638` căn)**: Đạt độ chính xác cấp độ sản phẩm thương mại cao với **`MAPE = 9.18%`**, **`MdAPE = 6.65%`**, **`PPE15 = 83.56%`**, **`PPE20 = 90.57%`**, sai số tuyệt đối trung vị chỉ **`361 triệu VNĐ`**, $R^2_{\log} = 0.933$!
    - **`Nhà` thổ cư (`7,860` căn)**: Đạt **`MAPE = 17.62%`**, **`MdAPE = 12.69%`**, **`PPE20 = 68.79%`**, $R^2_{\log} = 0.890$ — mức rất tốt đối với thị trường nhà ngõ Hà Nội nơi hai căn nhà cùng diện tích cách nhau 30m (một căn ngõ ô tô, một căn ngõ xe máy cụt) có thể chênh giá gấp đôi!

2. **Theo Phân khúc Giá (`price_tier`)**:
    - Mô hình hoạt động chính xác nhất ở dải thanh khoản chủ đạo của thị trường Hà Nội: **`3 – 6 tỷ` (`MAPE = 11.05%`, `MdAPE = 7.09%`, `PPE20 = 86.29%`)** và **`6 – 12 tỷ` (`MAPE = 13.43%`, `MdAPE = 9.51%`, `PPE20 = 79.67%`)** — hai phân khúc này chiếm tới **62.6%** tổng lượng giao dịch toàn thành phố!
    - Ở phân khúc siêu sang **`> 60 tỷ`** (`248` tòa nhà/khách sạn/biệt thự lớn), `MdAPE` tăng lên `22.55%` do tính độc bản kiến trúc, vị trí góc phố cổ và thương hiệu khai thác dòng tiền không thể phản ánh hết qua các trường cơ bản.

3. **Theo Địa bàn Quận/Huyện tiêu biểu (trong `reports/slice_error_analysis.csv`)**:
    - Các quận/huyện có tỷ trọng khu đô thị quy hoạch đồng bộ cao đạt sai số cực thấp: **Gia Lâm** (`MdAPE = 5.77%`, `PPE20 = 95.33%`), **Hoài Đức** (`MdAPE = 8.10%`), **Hà Đông** (`MdAPE = 8.20%`, `PPE20 = 82.19%`), **Nam Từ Liêm** (`MdAPE = 8.43%`, `PPE20 = 83.42%`), **Hoàng Mai** (`MdAPE = 8.49%`, `PPE20 = 80.04%`), **Bắc Từ Liêm** (`MdAPE = 9.97%`, `PPE20 = 81.32%`), **Thanh Xuân** (`MdAPE = 10.41%`).
    - Các quận lõi lịch sử có độ phân hóa ngõ nhỏ vs mặt phố cổ cực lớn như **Cầu Giấy** (`MdAPE = 11.63%`), **Ba Đình** (`MdAPE = 11.79%`), **Đống Đa** (`MdAPE = 12.43%`), **Tây Hồ** (`MdAPE = 13.07%`), **Hoàn Kiếm** (`MdAPE = 16.36%`) có sai số cao hơn tương ứng với phương sai thực tế của thị trường.

4. **Giải mã Nghịch lý Simpson (Simpson's Paradox) ở lát cắt `info_completeness`**:
    - Nhìn thoáng qua bảng tổng hợp: nhóm `2_Medium_2_3_attrs` có `MdAPE = 8.81%`, thấp hơn nhóm `3_High_4_5_attrs` (`MdAPE = 12.33%`). Tại sao nhiều thuộc tính hơn lại có sai số trung vị lớn hơn?
    - **Bản chất thống kê (Biến nhiễu / Confounder `property_type`)**: Gần như toàn bộ `Căn hộ chung cư` chỉ có đúng 2 thuộc tính vật lý là `bedroom_count` và `bathroom_count` (không có số tầng nhà, mặt tiền đất, độ rộng ngõ) nên rơi trọn vào nhóm `2_Medium_2_3_attrs`, kéo sai số nhóm này xuống `8.81%`. Trong khi đó, nhóm `3_High_4_5_attrs` chứa `99%` là `Nhà` thổ cư! Khi so sánh công bằng cùng loại `Nhà`, nhóm `1_Low_0_1_attrs` có sai số cao nhất (`MAPE = 19.00%`, `MdAPE = 13.29%`) và giảm rõ rệt khi có đầy đủ thông tin (`MAPE = 16.77%`, `MdAPE = 12.33%`)!

#### 3.3. Đối sánh Độ Quan Trọng Đặc Trưng (`% Gain`) giữa Hai Chuyên Gia LightGBM

| Hạng | Đặc trưng                 | Unified Gain (%) | **House Expert Gain (%)** | **Apartment Expert Gain (%)** | Ý nghĩa Nghiệp vụ BĐS Hà Nội                                                                          |
| :--: | :------------------------ | :--------------: | :-----------------------: | :---------------------------: | :---------------------------------------------------------------------------------------------------- |
|  1   | `te_project_log_ppm2`     |      35.02%      |           3.12%           |          **61.09%**           | Tên dự án chung cư quyết định hơn 61% phương sai đơn giá căn hộ!                                      |
|  2   | `te_ward_log_ppm2`        |      29.83%      |           4.41%           |             0.79%             | Mặt bằng giá cấp Phường giữ vai trò cầu nối vĩ mô khi gộp chung.                                      |
|  3   | `te_street_log_ppm2`      |      7.45%       |        **25.00%**         |             1.46%             | Tuyến phố là yếu tố số 1 quyết định đơn giá đất của `Nhà` thổ cư!                                     |
|  4   | `spatial_anchor_log_ppm2` |      10.25%      |        **15.11%**         |          **11.20%**           | Mỏ neo không gian kết hợp Empirical Bayes phát huy sức mạnh ở cả hai nhánh.                           |
|  5   | `floor_count_eff`         |      2.44%       |         **7.06%**         |             0.49%             | Số tầng (khôi phục 27,056 dòng từ văn bản!) quyết định giá trị xây dựng của `Nhà`.                    |
|  6   | `access_tier`             |      1.44%       |         **6.00%**         |             0.26%             | Cấp độ tiếp cận (ngõ nhỏ $\rightarrow$ ô tô tránh $\rightarrow$ mặt phố) cực kỳ quan trọng với `Nhà`. |
|  7   | `is_kinh_doanh`           |      1.22%       |         **6.10%**         |             0.13%             | Khả năng kinh doanh thương mại tạo bước nhảy giá mạnh cho nhà thổ cư.                                 |
|  8   | `area_per_bedroom`        |      0.67%       |           1.59%           |           **2.31%**           | Tỷ lệ diện tích/phòng ngủ phân biệt căn hộ cao cấp phòng rộng với căn chia nhỏ.                       |
|  9   | `total_floor_area_est`    |      0.74%       |         **2.17%**         |             0.02%             | Tổng diện tích sàn xây dựng ước tính chỉ tác động lên `Nhà`.                                          |
|  10  | `shape_aspect_ratio`      |      0.58%       |         **1.70%**         |             0.01%             | Tỷ lệ mặt tiền/chiều sâu phản ánh độ vuông đẹp của thửa đất thổ cư.                                   |

---

## PHẦN VI: GIAI ĐOẠN 7 (P1) — MODULE/CLI ĐỊNH GIÁ THỰC TẾ KÈM GIẢI THÍCH TREESHAP & BỘ KIỂM THỬ TỰ ĐỘNG `PYTEST` TOÀN DIỆN

### 1. Giải thích Chi tiết File [predict_valuation.py](file:///d:/PROPERTY-VALUATION/predict_valuation.py)

Đây là cổng giao tiếp suy diễn (Inference Engine & CLI) sẵn sàng tích hợp vào API hoặc chạy trực tiếp từ dòng lệnh cho người dùng cuối:

1. **Hàm `compute_local_treeshap_drivers(system, X_feat_one, top_k=5)`**:
    - **Kiến thức ML — Exact TreeSHAP (`pred_contrib=True`)**: Khác với độ quan trọng toàn cục (`Gain` chỉ cho biết biến nào quan trọng trung bình trên toàn thành phố), thuật toán **TreeSHAP** (Lundberg & Lee, 2017) được tích hợp gốc bằng C++ trong LightGBM tính toán **giá trị Shapley chính xác** $\phi_j(x)$ cho **riêng căn nhà đang định giá**:
      $$\hat{y}_{\text{unit}}(x) = \phi_0 + \sum_{j=1}^{110} \phi_j(x)$$
    - Hàm phối hợp tuyến tính vector Shapley của nhánh `Segmented` và `Unified` theo đúng trọng số $w_{\text{seg}} = 0.70$, quy đổi đóng góp logarit $\phi_j$ sang phần trăm tác động lên đơn giá $(\exp(\phi_j) - 1) \times 100\%$, trả về Top 5 yếu tố kéo tăng/giảm giá mạnh nhất của chính tài sản đó!
2. **Hàm `valuate_single_property(system, payload)`**:
    - Nhận dictionary thông tin tài sản thô (tự động điền giá trị mặc định an toàn nếu người dùng chỉ nhập các trường tối thiểu như `district_name` và `area`), chạy qua `MondrianConformalValuationSystem` và trả về JSON có cấu trúc gồm: `fair_price_vnd`, `fair_price_ty`, `unit_price_million_per_m2`, khoảng 80% `[conformal_80_low_ty, conformal_80_high_ty]` và `top_treeshap_drivers`.
3. **Giao diện dòng lệnh `argparse` (`build_parser` & `main`)**:
    - Chế độ `--demo`: Định giá 4 kịch bản thực tế điển hình tại Hà Nội và lưu kết quả vào `reports/demo_valuations.json`.
    - Chế độ `--input-json <file.json>`: Định giá hàng loạt (batch) từ tệp JSON.
    - Chế độ tham số lẻ (`--type`, `--district`, `--ward`, `--street`, `--project`, `--area`, `--floors`, `--frontage`, `--road-width`, `--bedrooms`, `--bathrooms`, `--name`, `--description`).

#### Kết quả Định giá Thực tế 4 Kịch bản Điển hình (`predict_valuation.py --demo`)

| Kịch bản Bất động sản Thực tế tại Hà Nội                                                        | Giá Định Giá Hợp Lý |  Đơn giá / m²  |   Khoảng Tin Cậy 80% (`[P10, P90]`)   | Giải thích TreeSHAP Nổi bật                                                                                            |
| :---------------------------------------------------------------------------------------------- | :-----------------: | :------------: | :-----------------------------------: | :--------------------------------------------------------------------------------------------------------------------- |
| **1. Chung cư 2PN `55m²` Vinhomes Smart City** (_Sapphire 2, Tây Mỗ, Nam Từ Liêm_)              |   **`3.656 tỷ`**    | `66.47 tr/m²`  |  `[3.282 – 4.112 tỷ]` _(rộng 22.7%)_  | Định vị mặt bằng dự án Sapphire 2 & phường Tây Mỗ ổn định, khoảng giá rất hẹp.                                         |
| **2. Chung cư góc 3PN `85m²` Trần Duy Hưng** (_Trung Hòa, Cầu Giấy, view hồ_)                   |   **`6.698 tỷ`**    | `78.80 tr/m²`  |  `[5.463 – 8.115 tỷ]` _(rộng 39.6%)_  | Tín hiệu `is_view_dep = 1` cộng thưởng `+1.5%`; tự động fallback mượt về phường Trung Hòa & đường Trần Duy Hưng.       |
| **3. Nhà dân xây 5 tầng `38m²` ngõ 2.5m Kim Giang** (_Đại Kim, Hoàng Mai, MT 3.8m_)             |   **`7.200 tỷ`**    | `189.48 tr/m²` |  `[6.445 – 8.462 tỷ]` _(rộng 28.0%)_  | Vị trí Đại Kim kéo tăng (`+4.11%`), nhưng ngõ `2.5m` (`-4.22%`) và không kinh doanh (`-4.42%`) điều chỉnh giảm hợp lý. |
| **4. Nhà mặt phố Trung Kính `65m²` × 7 tầng thang máy** (_Yên Hòa, Cầu Giấy, đường 15m vỉa hè_) |   **`29.507 tỷ`**   | `453.95 tr/m²` | `[22.072 – 38.823 tỷ]` _(rộng 56.8%)_ | Tuyến phố Trung Kính (`+11.31%`), mỏ neo vị trí (`+8.26%`), kết cấu 7 tầng (`+7.91%`), đường 15m (`+7.07%`).           |

---

### 2. Giải thích Bộ Kiểm Thử Tự Động `pytest` (`13` Bài Kiểm Thử Toàn Diện)

Toàn bộ dự án được bảo vệ bởi **4 tệp kiểm thử tự động** trong thư mục `tests/`:

1. **[tests/test_raw_immutability.py](file:///d:/PROPERTY-VALUATION/tests/test_raw_immutability.py)** (`1` test):
    - `test_raw_files_exist_and_preserve_exact_row_counts`: Xác minh `data/raw/shard_0000.parquet` (`350,075 × 19`) và `data/raw/hanoi_housing_raw.parquet` (`93,940 × 19`) còn nguyên vẹn tuyệt đối.
2. **[tests/test_cleaning.py](file:///d:/PROPERTY-VALUATION/tests/test_cleaning.py)** (`5` tests):
    - Kiểm chứng chuẩn hóa hướng nhà, chuẩn hóa tên đường bảo toàn đường đánh số (`Đường 800A`), lọc lỗi đơn vị 1000x & không rò rỉ cột tạm `ppm2`, sanitize mặt tiền phi vật lý mà không mất dòng, và khử chính xác tin repost môi giới.
3. **[tests/test_no_leakage.py](file:///d:/PROPERTY-VALUATION/tests/test_no_leakage.py)** (`3` tests):
    - `test_gold_splits_strict_chronology_and_zero_group_overlap`: Chứng minh $\max(t_{\text{train}}) < \min(t_{\text{val}}) < \max(t_{\text{val}}) < \min(t_{\text{test}})$ và giao các tập `property_group_id` bằng rỗng.
    - `test_nlp_feature_extraction_is_invariant_to_asking_price_mentions`: Chứng minh thay đổi giá bán trong tiêu đề/mô tả không làm thay đổi dù chỉ 1 bit trong vector đặc trưng.
    - `test_oof_target_encoder_does_not_leak_own_row_or_test_set`: Chứng minh bộ mã hóa không gian OOF không nhìn thấy nhãn của chính dòng đang huấn luyện và không có bất kỳ cột nào chứa `"price"`.
4. **[tests/test_models_and_inference.py](file:///d:/PROPERTY-VALUATION/tests/test_models_and_inference.py)** (`4` tests):
    - `test_champion_model_significantly_beats_naive_baseline_on_test`: Xác minh mô hình vô địch vượt xa Baseline trên mọi chỉ số (`MAPE`, `MdAPE`, `PPE20`, $R^2_{\log}$).
    - `test_conformal_intervals_non_crossing_and_calibrated_coverage`: Xác minh 100% dự đoán thỏa `P10 < Point < P90` và độ phủ thực tế nằm trong vùng hiệu chỉnh chuẩn quanh 80%.
    - `test_economic_monotonicity_area_and_streetfront_premium`: Kiểm tra quy luật kinh tế học bất động sản (_ceteris paribus_): tăng diện tích `40m² → 60m² → 85m²` giá định giá tăng đơn điệu ngặt; nâng cấp từ nhà ngõ nhỏ lên nhà mặt phố vỉa hè thang máy cùng diện tích tại Trung Kính làm giá trị tăng vượt trội `> 25%`!
    - `test_cli_entrypoint_single_property_flags`: Gọi trực tiếp tiến trình con `predict_valuation.py` với cờ dòng lệnh định giá căn nhà phân lô phố Thái Hà (`48m² × 5 tầng`) và xác minh JSON đầu ra hợp lệ.

---

### 3. Cấu Trúc Thư Mục Hoàn Chỉnh & Hướng Dẫn Chạy Toàn Trình (End-to-End Commands)

```
D:\PROPERTY-VALUATION\
├── data/
│   ├── raw/                                  # [BẤT BIẾN - READ ONLY]
│   │   ├── shard_0000.parquet                # 350,075 dòng × 19 cột (Toàn quốc)
│   │   └── hanoi_housing_raw.parquet         # 93,940 dòng × 19 cột (Nhà & Chung cư Hà Nội)
│   ├── silver/                               # [TẦNG LÀM SẠCH TRUNG GIAN]
│   │   ├── hanoi_missing_handled.parquet     # 90,938 dòng × 25 cột (Xử lý missing cơ bản)
│   │   ├── hanoi_deduplicated.parquet        # 90,073 dòng × 25 cột (Khử trùng hoàn toàn)
│   │   └── hanoi_cleaned.parquet             # 77,653 dòng × 26 cột (Chuẩn hóa, lọc ngoại lai & khử repost)
│   └── gold/                                 # [TẦNG HUẤN LUYỆN CHỐNG LEAKAGE]
│       ├── train.parquet                     # 54,357 dòng (01/06/2025 -> 28/06/2025)
│       ├── val.parquet                       # 11,480 dòng (28/06/2025 -> 14/07/2025, purged)
│       └── test.parquet                      # 11,498 dòng (14/07/2025 -> 30/07/2025, purged)
├── models/
│   ├── champion_point_bundle.joblib          # Feature Pipeline + Blended Segmented/Unified LightGBM
│   └── avm_conformal_system.joblib           # Hệ thống AVM hoàn chỉnh (Point + Mondrian CQR [P10, P90])
├── reports/
│   ├── duplicate_candidates.csv              # Báo cáo nhóm nghi trùng (Giai đoạn 2B)
│   ├── removed_duplicates.csv                # 865 dòng trùng hoàn toàn bị loại (Giai đoạn 2B)
│   ├── outlier_and_repost_report.csv         # 12,420 dòng ngoại lai & repost bị loại (Giai đoạn 3)
│   ├── cleaning_summary.json                 # Thống kê kiểm toán làm sạch (Giai đoạn 3)
│   ├── split_summary.json                    # Thống kê chia tập & NLP backfill (Giai đoạn 4)
│   ├── model_comparison.csv                  # Bảng so sánh 6 tầng mô hình trên Val & Test (Giai đoạn 5)
│   ├── model_training_summary.json           # Chi tiết siêu tham số & 95% Bootstrap CI (Giai đoạn 5)
│   ├── slice_error_analysis.csv              # Phân tích lỗi theo loại BĐS, giá, quận, độ đầy đủ (Giai đoạn 6)
│   ├── feature_importance.csv                # Độ quan trọng % Gain của 110 đặc trưng (Giai đoạn 6)
│   ├── conformal_and_slice_summary.json      # Kết quả hiệu chỉnh CQR 80% (Giai đoạn 6)
│   └── demo_valuations.json                  # Kết quả định giá + TreeSHAP cho 4 ca thực tế (Giai đoạn 7)
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                      # Logic làm sạch chuyên sâu, lọc ngoại lai & tạo property_group_id
│   ├── splitting.py                          # Logic Purged Chronological Walk-Forward Split
│   ├── features.py                           # Price-Redacted NLP + 5-Fold OOF Hierarchical Empirical Bayes TE
│   ├── modeling.py                           # 5 tầng mô hình + Ensemble + Metrics AVM + Bootstrap CI
│   └── intervals_and_analysis.py             # Mondrian Split CQR [P10, P90] + Slice Analysis + Gain Importance
├── tests/
│   ├── test_raw_immutability.py              # Kiểm chứng bất biến dữ liệu Raw
│   ├── test_cleaning.py                      # Kiểm chứng logic làm sạch & khử repost
│   ├── test_no_leakage.py                    # Kiểm chứng toán học Zero Data Leakage
│   └── test_models_and_inference.py          # Kiểm chứng chất lượng mô hình, CQR, đơn điệu kinh tế & CLI
├── inspect_data.py                           # Script Giai đoạn 1A (đã có)
├── filter_hanoi.py                           # Script Giai đoạn 1B (đã có — không chạy đè lại Raw)
├── check_missing.py                          # Script Giai đoạn 2A (đã có)
├── handle_missing.py                         # Script Giai đoạn 2A (đã có)
├── check_duplicates.py                       # Script Giai đoạn 2B (đã có)
├── remove_duplicates.py                      # Script Giai đoạn 2B (đã có)
├── clean_outliers.py                         # Script chạy Giai đoạn 3 (P0)
├── split_dataset.py                          # Script chạy Giai đoạn 4 (P0)
├── train_models.py                           # Script chạy Giai đoạn 5 (P0)
├── analyze_and_calibrate.py                  # Script chạy Giai đoạn 6 (P1)
├── predict_valuation.py                      # CLI định giá thực tế & giải thích TreeSHAP Giai đoạn 7 (P1)
├── requirements.txt                          # Danh sách phiên bản thư viện khóa chặt trong .venv
└── HUONG_DAN_DU_AN_TUNG_BUOC.md              # Tài liệu hướng dẫn & kiểm toán toàn diện dự án
```

#### Các lệnh vận hành chuẩn từ đầu đến cuối (trên môi trường `.venv`):

```powershell
# 1. Làm sạch chuyên sâu & khử near-repost (Silver -> Cleaned Silver)
.\.venv\Scripts\python.exe clean_outliers.py

# 2. Chia tập Purged Chronological Split & kiểm tra pipeline đặc trưng (Cleaned Silver -> Gold)
.\.venv\Scripts\python.exe split_dataset.py

# 3. Huấn luyện hệ thống 6 phương án mô hình & tính 95% Bootstrap CI trên Test
.\.venv\Scripts\python.exe train_models.py

# 4. Hiệu chỉnh khoảng giá 80% Mondrian CQR, phân tích lỗi theo lát cắt & Feature Importance
.\.venv\Scripts\python.exe analyze_and_calibrate.py

# 5. Chạy định giá thử nghiệm 4 kịch bản Hà Nội kèm giải thích TreeSHAP
.\.venv\Scripts\python.exe predict_valuation.py --demo

# 6. Định giá bất kỳ một bất động sản cụ thể từ dòng lệnh
.\.venv\Scripts\python.exe predict_valuation.py --type "Nhà" --district "Đống Đa" --ward "Láng Hạ" --street "Thái Hà" --area 48 --floors 5 --frontage 4.2 --road-width 4.5 --bedrooms 4 --bathrooms 4 --name "Nhà phân lô Thái Hà ô tô vào nhà 5 tầng thang máy"

# 7. Chạy toàn bộ 13 bài kiểm thử tự động (Raw Immutability + Cleaning + Anti-Leakage + AVM & CLI)
.\.venv\Scripts\python.exe -m pytest -v
```

#### Kết quả chạy `pytest` cuối cùng toàn dự án:

- **`13 passed in 7.98s` — 100% PASS, 0 lỗi, 0 cảnh báo!**

Quy trình 3 lớp phòng thủ
77.653 bản ghi sạch
│
▼
[LỚP 1 — CHỐNG LEAKAGE THỜI GIAN]
Chronological Split + Group Purging
│
▼
Train / Validation / Test
│
▼
[LỚP 2 — CHỐNG TEXT LEAKAGE]
Che giá trong name/description
│
▼
Trích xuất thuộc tính hợp lệ
│
▼
[LỚP 3 — CHỐNG TARGET LEAKAGE]
OOF Target Encoding + Hierarchical Shrinkage
│
▼
Feature Matrix
│
▼
Giai đoạn 5: Training Model

1. Phương án triển khai
   Thuật toán Trạng thái Hướng xử lý
   Ridge Regression Đã hoàn thành Giữ nguyên
   Histogram Gradient Boosting Đã hoàn thành Giữ nguyên
   LightGBM Unified Đã hoàn thành Giữ nguyên
   LightGBM Segmented Đã hoàn thành Giữ nguyên
   LightGBM Ensemble Đã hoàn thành Giữ làm mô hình chính
   Decision Tree Regressor (L3) Chưa có Bổ sung
   MLP Regressor (L6) Chưa có Bổ sung

1. Hai thuật toán nên triển khai
1. Decision Tree Regressor
   Ưu tiên 1

Liên hệ: L3 – Decision Trees.

Sử dụng cây quyết định để dự đoán giá nhà thông qua diện tích, vị trí, số tầng, mặt tiền...

Dễ triển khai với dữ liệu hiện tại và dễ giải thích khi bảo vệ.

Lưu ý: L3 dạy trực tiếp ID3/C4.5 cho phân loại. Decision Tree Regressor là biến thể hồi quy của cùng họ thuật toán, nên cần được giảng viên chấp nhận.

2. Neural Network – MLP Regressor
   Ưu tiên 2

Liên hệ: L6 – Neural Network.

Xây dựng mạng nơ-ron nhiều lớp để dự đoán giá bất động sản.

Sử dụng các kiến thức được giảng dạy như Perceptron, Gradient Descent và Backpropagation.

Đây là lựa chọn phù hợp vì bài toán của bạn là hồi quy, không phải phân loại.

File split_dataset.py là chương trình điều phối Giai đoạn 4 trong dự án.

Đầu tiên, em đọc dữ liệu đã làm sạch từ Silver Layer, gồm 77.653 bản ghi theo báo cáo.

Sau đó, em gọi hàm purged_chronological_split() để chia dữ liệu theo thời gian với tỷ lệ dự kiến 70% Train, 15% Validation và 15% Test, đồng thời loại các bản ghi ở tập sau có mã property_group_id đã xuất hiện trong các tập trước.

Tiếp theo, em lưu ba tập dữ liệu thành các file Parquet trong thư mục data/gold.

Sau khi chia, em khởi tạo HanoiValuationFeaturePipeline, dùng fit_transform() trên Train và transform() trên Validation, Test.

Feature Pipeline tạo các đặc trưng vật lý, hình học, thông tin văn bản và thống kê vị trí. Theo kết quả của dự án, mỗi bản ghi được biểu diễn bằng 110 đặc trưng số.

Cuối cùng, em đếm số giá trị vật lý được khôi phục từ văn bản, ghi nhận số lượng feature và kích thước các ma trận đầu ra, sau đó lưu vào reports/split_summary.json.

File này chưa huấn luyện mô hình dự đoán giá mà mới chia dữ liệu và kiểm chứng Feature Engineering.

Kết luận quan trọng nhất: Trong split_dataset.py, dữ liệu được chia thành ba tập trước, rồi HanoiValuationFeaturePipeline mới được fit trên Train và dùng lại để Transform Validation/Test. Bước này chuẩn bị và kiểm tra dữ liệu cho Giai đoạn 5, chứ không tạo ra mô hình dự đoán giá hoàn chỉnh.
