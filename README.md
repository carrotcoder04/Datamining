# Customer Segmentation - Data Mining Project

Dự án Khai phá Dữ liệu (Data Mining) áp dụng kỹ thuật **RFM (Recency, Frequency, Monetary)** kết hợp với các thuật toán Học máy không giám sát (K-Means, Hierarchical Clustering) để phân khúc khách hàng.

## 📌 Tổng quan Pipeline
Quy trình của dự án được chia thành 4 giai đoạn chính:
1. **Làm sạch dữ liệu (Data Cleaning):** Xử lý giá trị thiếu, loại bỏ dữ liệu bất thường và trùng lặp.
2. **Trích xuất đặc trưng (Feature Engineering):** Tính toán chỉ số R, F, M từ lịch sử giao dịch mua hàng.
3. **Chuẩn hóa dữ liệu (Scaling):** Đưa các đặc trưng về cùng một không gian tỉ lệ.
4. **Phân nhóm khách hàng (Clustering):** Gom cụm dữ liệu thành 3 phân khúc khách hàng chuyên biệt và trực quan hóa kết quả.

## ⚠️ Lưu ý quan trọng về Checkpoint
Trong các script gom cụm (`kmeans_algorithms.py` và `hierachical_cluster.py`) có sử dụng cơ chế **Checkpoint**. Nó sẽ lưu lại mô hình (file `.pkl`, `.npy`) trong lần chạy đầu tiên để các lần sau chạy nhanh hơn.

**Nếu bạn có thay đổi số lượng cụm (clusters) hoặc thay đổi dữ liệu đầu vào**, bạn **BẮT BUỘC phải xóa các file mô hình cũ** nằm trong thư mục trước khi chạy, nếu không code sẽ tiếp tục sử dụng mô hình sai cũ:
- Xóa `kmeans_model.pkl` (đối với mô hình K-Means)
- Xóa `hierarchical_labels.npy` (đối với mô hình Hierarchical)

---

## 🚀 Hướng dẫn chạy dự án

### Cài đặt thư viện
Đảm bảo bạn đã cài đặt các thư viện cần thiết. Mở Terminal / Command Prompt và chạy lệnh sau:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### Chạy tuần tự các bước
*(Mở Terminal tại thư mục gốc của project)*

#### Bước 1: Làm sạch dữ liệu (Data Cleaning)
Chạy lần lượt các file sau để lọc rác, điểm dị thường và xử lý lỗi từ dữ liệu thô:
```bash
python checkNaN.py
python drop_miss_rows.py
python drop_negative_quantity.py
python CheckDuplicated.py
python remove_outliner.py
```
👉 *Kết quả:* Xuất ra file dữ liệu sạch `clean_data.csv`.

#### Bước 2: Tạo bộ dữ liệu RFM
```bash
python Create_RFM.py
```
👉 *Kết quả:* Tính toán và xuất ra file `rfm_data.csv`.

#### Bước 3: Chuẩn hóa dữ liệu (Data Scaling)
Đưa các giá trị R, F, M về cùng một tỉ lệ để thuật toán gom cụm hoạt động chính xác.
```bash
python scaled_data.py
```
👉 *Kết quả:* Xuất ra file `scaled_data.csv`.

#### Bước 4: Phân nhóm khách hàng (Clustering)
Bạn có thể chạy thuật toán dưới đây (Mô hình đã được cấu hình mặc định chia thành **3 phân khúc khách hàng**):

**Tùy chọn A - Chạy thuật toán K-Means Clustering:**
```bash
python kmeans_algorithms.py
```
*Kết quả sinh ra:*
- Bảng dữ liệu đã gắn nhãn cụm: `scaled_data_kmeans.csv`
- Bảng thống kê trung bình của từng nhóm: `final_data_kmeans_category.csv`
- Biểu đồ trực quan: `kmeans_3d_plot.png`, `kmeans_pie_chart.png`

**Tùy chọn B - Chạy thuật toán Hierarchical Clustering:**
```bash
python hierachical_cluster.py
```
*Kết quả sinh ra:*
- Bảng dữ liệu đã gắn nhãn cụm: `scaled_data_hierarchical.csv`
- Bảng thống kê trung bình của từng nhóm: `final_data_hierarchical_category.csv`
- Biểu đồ trực quan: `hierarchical_3d_plot.png`, `hierarchical_pie_chart.png`
