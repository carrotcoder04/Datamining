import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("rfm_data.csv")

# =========================
# 2. KIỂM TRA TRÙNG
# =========================
duplicate_count = df["CustomerID"].duplicated().sum()

print("Số lượng CustomerID bị trùng:", duplicate_count)

# =========================
# 3. HIỂN THỊ CÁC DÒNG TRÙNG
# =========================
duplicates = df[df["CustomerID"].duplicated(keep=False)]

print("\nCác CustomerID bị trùng:")
print(duplicates.head())

# =========================
# 4. SỐ CUSTOMER DUY NHẤT
# =========================
unique_count = df["CustomerID"].nunique()

print("\nSố CustomerID duy nhất:", unique_count)
print("Tổng số dòng:", len(df))