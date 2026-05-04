import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("rfm_data.csv")

cols = ["Recency", "Frequency", "Monetary"]

# =========================
# 2. FIGURE 4K
# =========================
plt.figure(figsize=(16, 9), dpi=300)  # 4K gần chuẩn

# =========================
# 3. HISTOGRAM
# =========================
for i, col in enumerate(cols):
    plt.subplot(2, 2, i+1)
    df[col].hist(bins=50)
    plt.title(f"{col} Distribution", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")

# =========================
# 4. BOXPLOT CHUNG
# =========================
plt.subplot(2, 2, 4)
df[cols].boxplot()
plt.title("Outliers Detection (Boxplot)", fontsize=14)

# =========================
# 5. SAVE 4K
# =========================
plt.tight_layout()
plt.savefig("rfm_4k.png", dpi=300)  # QUAN TRỌNG
plt.show()

print("Đã lưu ảnh 4K: rfm_4k.png")