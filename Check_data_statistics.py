import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("drop_miss_row.csv")

# =========================
# 2. THỐNG KÊ TỔNG HỢP
# =========================
# include='all' để lấy cả numeric + object
stats = df.describe(include='all').transpose()

# =========================
# 3. THÊM UNIQUE (vì describe chưa có)
# =========================
stats["unique"] = df.nunique()

# =========================
# 4. SẮP XẾP CỘT CHO ĐẸP
# =========================
stats = stats[[
    "count", "unique", "top", "freq",
    "mean", "std", "min", "25%", "50%", "75%", "max"
]]

# =========================
# 5. HIỂN THỊ
# =========================
print("\n=== DATA STATISTICS ===")
print(stats)

# =========================
# 6. LƯU FILE (CHO BÁO CÁO)
# =========================
stats.to_csv("data_statistics.csv")

print("\nĐã lưu file: data_statistics.csv")