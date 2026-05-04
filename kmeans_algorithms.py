import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import joblib
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "kmeans_model.pkl"

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("scaled_data.csv")
X = df[["Recency", "Frequency", "Monetary"]].values

# =========================
# 2. KMEANS CLUSTERING (CHECKPOINT)
# =========================
if os.path.exists(MODEL_PATH):
    print("🔁 Loading KMeans model...")
    kmeans = joblib.load(MODEL_PATH)
    labels = kmeans.predict(X)
else:
    print("🚀 Training KMeans...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    joblib.dump(kmeans, MODEL_PATH)
    print("✅ Model saved!")

# Mapping label
label_map = {0: "I", 1: "II", 2: "III"}
df["Cluster_Label"] = [label_map[l] for l in labels]

# =========================
# 3. SAVE DATA
# =========================
df.to_csv("scaled_data_kmeans.csv", index=False)

# =========================
# 4. CATEGORY (DATA GỐC)
# =========================
df_original = pd.read_csv("final_data.csv")

df_merged = df_original.merge(
    df[["CustomerID", "Cluster_Label"]],
    on="CustomerID",
    how="left"
)

summary = df_merged.groupby("Cluster_Label").agg({
    "CustomerID": "count",
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
}).rename(columns={"CustomerID": "Count"}).reset_index()

summary.to_csv("final_data_kmeans_category.csv", index=False)

# =========================
# 5. 3D PLOT (4K)
# =========================
fig = plt.figure(figsize=(16, 9), dpi=240)
ax = fig.add_subplot(111, projection='3d')

colors = ["red", "blue", "green"]

for i in range(3):
    idx = labels == i
    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2],
               label=label_map[i], color=colors[i], s=20)

ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary")
ax.set_title("KMeans Clustering (3D)")
ax.legend()

plt.savefig("kmeans_3d_plot.png", dpi=240)
plt.close()

# =========================
# 6. PIE CHART
# =========================
counts = df["Cluster_Label"].value_counts().sort_index()

plt.figure(figsize=(16, 9), dpi=240)
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("KMeans Cluster Distribution")

plt.savefig("kmeans_pie_chart.png", dpi=240)
plt.close()

print("✅ KMeans done!")