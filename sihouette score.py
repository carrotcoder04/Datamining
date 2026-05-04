import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("scaled_data.csv")
X = df[["Recency", "Frequency", "Monetary"]]

# =========================
# 2. (KHÔNG CẦN SCALE LẠI)
# =========================
X_scaled = X.values  # đã scale rồi

# =========================
# 3. CONFIG
# =========================
k_start = 2
k_end = 30
step = 1   # ⚠️ KHÔNG nên =1 nếu chạy tới 1000

k_values = list(range(k_start, k_end + 1, step))

scores = []

# =========================
# 4. SILHOUETTE (TỐI ƯU)
# =========================
for k in k_values:
    if k >= len(X_scaled):
        break  # tránh lỗi

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # ⚠️ dùng sample để tăng tốc
    score = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
    scores.append(score)

    print(f"k = {k}, silhouette = {score:.4f}")

# =========================
# 5. VẼ ẢNH 8K
# =========================
plt.figure(figsize=(32, 18), dpi=250)  # ~8K (7680x4320)

plt.plot(k_values[:len(scores)], scores, marker='o', linewidth=2)

plt.xlabel("Number of clusters (k)", fontsize=20)
plt.ylabel("Silhouette Score", fontsize=20)
plt.title("Silhouette Score vs k (High Resolution)", fontsize=24)

plt.grid()

# highlight best k
best_k = k_values[scores.index(max(scores))]
best_score = max(scores)

plt.scatter(best_k, best_score)
plt.text(best_k, best_score, f"  Best k={best_k}", fontsize=16)

plt.tight_layout()
plt.savefig("silhouette_4k_2_30.png", dpi=250)
plt.show()

print("\nBest k:", best_k)