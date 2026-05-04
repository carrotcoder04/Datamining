import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_data.csv")

X = df[["Recency", "Frequency", "Monetary"]]

# =========================
# 2. SCALE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. PARAMETER
# =========================
min_k = 2
max_k = 14
step = 1

k_values = list(range(min_k, max_k + 1, step))
scores = []

# =========================
# 4. TÍNH SILHOUETTE
# =========================
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    scores.append(score)

    print(f"k = {k}, silhouette = {score:.4f}")

# =========================
# 5. VẼ ẢNH 4K
# =========================
plt.figure(figsize=(12.8, 7.2), dpi=300)  # ~3840x2160

plt.plot(k_values, scores, marker='o')
plt.xlabel("Number of clusters (k)", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.title("Silhouette Score vs K", fontsize=16)
plt.grid()

# highlight best k
best_k = k_values[scores.index(max(scores))]
best_score = max(scores)

plt.scatter(best_k, best_score)
plt.text(best_k, best_score, f"  Best k={best_k}", fontsize=12)

# =========================
# 6. SAVE 4K
# =========================
plt.tight_layout()
plt.savefig("silhouette_4k.png", dpi=300)
plt.show()

print("\nBest k:", best_k)
print("Đã lưu ảnh: silhouette_4k.png")