import pandas as pd

# =========================
# 1. LOAD CSV
# =========================
df = pd.read_csv("raw_data.csv")

# =========================
# 2. KIỂM TRA TRƯỚC
# =========================
print("Shape trước:", df.shape)
print("Missing CustomerID:", df["CustomerID"].isnull().sum())

# =========================
# 3. DROP NULL
# =========================
df_clean = df.dropna(subset=["CustomerID", "Description"])

# =========================
# 4. KIỂM TRA SAU
# =========================
print("Shape sau:", df_clean.shape)
print("Missing CustomerID sau:", df_clean["CustomerID"].isnull().sum())
print("Missing Description sau:", df_clean["Description"].isnull().sum())

# =========================
# 5. LƯU FILE CLEAN
# =========================
df_clean.to_csv("drop_miss_row.csv", index=False)

print("Đã lưu: drop_miss_row.csv")