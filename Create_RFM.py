import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("clean_data.csv")

print("Shape:", df.shape)

# =========================
# 2. CHUYỂN ĐỔI KIỂU DỮ LIỆU
# =========================
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# =========================
# 3. TẠO TOTAL PRICE
# =========================
df["TotalPrice"] = df["Quantity"] * df["Price"]

# =========================
# 4. XÓA CustomerID NULL (nếu còn)
# =========================
df = df.dropna(subset=["CustomerID"])

# =========================
# 5. TÍNH RECENCY
# =========================
# - Tìm thời gian cuối cùng có giao dịch và gán ngày sau cuối cùng đó 1 ngày vào biến now.
now = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# - Tạo 1 cột mới có tên là “date” chỉ chứa duy nhất ngày lập hóa đơn.
df["date"] = df["InvoiceDate"].dt.date

# - Nhóm theo khách hàng và kiểm tra ngày mua cuối cùng sau đó lưu vào cột "LastPurshaceDate".
recency_df = df.groupby("CustomerID", as_index=False)["date"].max()
recency_df.columns = ["CustomerID", "LastPurshaceDate"]

# - Tính toán số ngày mua gần đây và lưu vào cột "Recency"
recency_df["Recency"] = recency_df["LastPurshaceDate"].apply(lambda x: (now.date() - x).days)
recency_df = recency_df.drop(columns=["LastPurshaceDate"])

# =========================
# 6. TÍNH FREQUENCY
# =========================
# - Loại bỏ các bản ghi có cùng mã hóa đơn và ID khách hàng
freq_df = df.drop_duplicates(subset=["Invoice", "CustomerID"])[["Invoice", "CustomerID"]]
# - Tính toán tần suất mua hàng cho mỗi khách hàng.
freq_df = freq_df.groupby("CustomerID", as_index=False)["Invoice"].count()
freq_df.columns = ["CustomerID", "Frequency"]

# =========================
# 7. TÍNH MONETARY
# =========================
# - Tính toán tổng tiền đã chi tiêu cho mỗi khách hàng.
monetary_df = df.groupby("CustomerID", as_index=False)["TotalPrice"].sum()
monetary_df.columns = ["CustomerID", "Monetary"]

# =========================
# 8. HỢP NHẤT DỮ LIỆU
# =========================
rfm = recency_df.merge(freq_df, on="CustomerID").merge(monetary_df, on="CustomerID")

# =========================
# 9. KIỂM TRA
# =========================
print("\nRFM sample:")
print(rfm.head())

print("\nShape RFM:", rfm.shape)

# =========================
# 8. LƯU FILE
# =========================
rfm.to_csv("rfm_data.csv", index=False)

print("\nĐã lưu file: rfm_data.csv")