import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("drop_miss_row.csv")

print("Shape ban đầu:", df.shape)

# =========================
# 2. REMOVE QUANTITY <= 0
# =========================
print("Số dòng Quantity <= 0:", (df["Quantity"] <= 0).sum())

df = df[df["Quantity"] > 0]

print("Shape sau khi remove Quantity <= 0:", df.shape)

# =========================
# 3. LƯU FILE
# =========================
df.to_csv("clean_data.csv", index=False)

print("Đã lưu file: clean_data.csv")