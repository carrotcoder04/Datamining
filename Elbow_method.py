import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("scaled_data.csv")
X = df[["Recency", "Frequency", "Monetary"]]

# =========================
# 2. KHÔNG SCALE LẠI
# =========================
X_scaled = X.values

# =========================
# 3. CONFIG
# =========================
k_start = 2
k_end = 30
step = 1

k_values = list(range(k_start, k_end + 1, step))

sse = []

# =========================
# 4. ELBOW METHOD
# =========================
for k in k_values:
    if k >= len(X_scaled):
        break

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    inertia = kmeans.inertia_
    sse.append(inertia)

    print(f"k = {k}, SSE = {inertia:.2f}")

# =========================
# 5. VẼ ẢNH 8K
# =========================
plt.figure(figsize=(32, 18), dpi=250)

plt.plot(k_values[:len(sse)], sse, marker='o', linewidth=2)

plt.xlabel("Number of clusters (k)", fontsize=20)
plt.ylabel("SSE (Inertia)", fontsize=20)
plt.title("Elbow Method For Optimal k (High Resolution)", fontsize=24)

plt.grid()

# =========================
# 6. TÌM "ELBOW" (approx)
# =========================
# dùng đạo hàm bậc 2 để tìm điểm gãy
import numpy as np

k_array = np.array(k_values[:len(sse)])
sse_array = np.array(sse)

# tính độ cong (second derivative)
curvature = np.diff(sse_array, 2)

elbow_index = np.argmax(-curvature) + 1
elbow_k = k_array[elbow_index]

# highlight
plt.scatter(elbow_k, sse_array[elbow_index])
plt.text(elbow_k, sse_array[elbow_index], f"  Elbow k={elbow_k}", fontsize=16)

plt.tight_layout()
plt.savefig("elbow_4k_2_30.png", dpi=250)
plt.show()

print("\nElbow k (approx):", elbow_k)