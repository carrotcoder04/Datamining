import pandas as pd
from validators import validate_create_rfm_input

# =========================
# 1. LOAD DATA
# =========================
# NOTE: this script now validates the input schema before processing.

df = pd.read_csv("clean_data.csv")

# Validate and normalize input (will raise ValueError with actionable message if invalid)
df = validate_create_rfm_input(df)

print("Shape:", df.shape)

# =========================
# 2. CHUYỂN ĐỔI KIỂU DỮ LIỆU
# =========================
# InvoiceDate already coerced in validator, Quantity and Price coerced as numeric

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
now = df["InvoiceDate"].max() + pd.Timedelta(days=1)

df["date"] = df["InvoiceDate"].dt.date

recency_df = df.groupby("CustomerID", as_index=False)["date"].max()
recency_df.columns = ["CustomerID", "LastPurshaceDate"]
recency_df["Recency"] = recency_df["LastPurshaceDate"].apply(lambda x: (now.date() - x).days)
recency_df = recency_df.drop(columns=["LastPurshaceDate"])

# =========================
# 6. TÍNH FREQUENCY
# =========================
freq_df = df.drop_duplicates(subset=["Invoice", "CustomerID"])[["Invoice", "CustomerID"]]
freq_df = freq_df.groupby("CustomerID", as_index=False)["Invoice"].count()
freq_df.columns = ["CustomerID", "Frequency"]

# =========================
# 7. TÍNH MONETARY
# =========================
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
# 10. LƯU FILE
# =========================
rfm.to_csv("rfm_data.csv", index=False)

print("\nĐã lưu file: rfm_data.csv")
