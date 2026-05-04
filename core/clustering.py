import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def train_kmeans(input_csv="scaled_data.csv"):
    print("🚀 Bắt đầu quá trình Phân cụm (Clustering)...")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file {input_csv}.")
        
    df = pd.read_csv(input_csv)
    X = df[["Recency", "Frequency", "Monetary"]].values
    
    # 1. Huấn luyện KMeans (k=3)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    df["Cluster"] = labels
    
    # 2. Đánh giá (Evaluation Metrics)
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    print(f"📈 Đánh giá Mô hình KMeans (k={k}):")
    print(f"   - Silhouette Score: {sil_score:.4f} (Càng gần 1 càng tốt)")
    print(f"   - Davies-Bouldin Index: {db_score:.4f} (Càng nhỏ càng tốt)")
    
    # 3. Ghi dữ liệu đã phân cụm
    df.to_csv("final_data_kmeans.csv", index=False)
    
    # Đọc lại dữ liệu gốc (trước khi scale) để lưu cùng cụm (dùng cho hiển thị trên Dashboard)
    rfm_df = pd.read_csv("rfm_data.csv")
    rfm_df["Cluster"] = labels
    rfm_df.to_csv("rfm_data_with_clusters.csv", index=False)
    
    # Tính trung bình theo cụm
    summary = rfm_df.groupby("Cluster").agg({
        "CustomerID": "count",
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    }).rename(columns={"CustomerID": "Count"}).reset_index()
    summary.to_csv("kmeans_cluster_summary.csv", index=False)
    
    # 4. Lưu mô hình
    joblib.dump(kmeans, "core/models/kmeans_model.pkl")
    print("✅ Đã phân cụm xong và lưu kmeans_model.pkl!")

if __name__ == "__main__":
    train_kmeans()
