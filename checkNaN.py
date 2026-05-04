import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("raw_data.csv")

# =========================
# 2. ĐẾM SỐ NaN
# =========================
missing_customer = df["CustomerID"].isnull().sum()
missing_desc = df["Description"].isnull().sum()

# =========================
# 3. TỶ LỆ %
# =========================
missing_percent_customer = (missing_customer / len(df)) * 100
missing_percent_desc = (missing_desc / len(df)) * 100

print("Số lượng CustomerID bị thiếu:", missing_customer, f"({round(missing_percent_customer, 2)}%)")
print("Số lượng Description bị thiếu:", missing_desc, f"({round(missing_percent_desc, 2)}%)")

# =========================
# 4. XEM MỘT VÀI DÒNG BỊ NaN
# =========================
print("\nMột số dòng bị thiếu CustomerID hoặc Description:")
print(df[df["CustomerID"].isnull() | df["Description"].isnull()].head())