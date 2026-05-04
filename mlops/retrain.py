import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from core.recommender import train_apriori

DB_PATH = "backend/feedback.db"
FEEDBACK_THRESHOLD = 50 

def get_feedbacks():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feedbacks", conn)
    conn.close()
    return df

def run_retraining():
    print("\n" + "="*50)
    print("🔄 BẮT ĐẦU QUÁ TRÌNH RETRAINING (MLOPS)")
    print("="*50)
    
    if not os.path.exists("rfm_data.csv"):
        print("⚠️ Không có dữ liệu gốc (rfm_data.csv). Hủy retrain.")
        return
        
    df_rfm = pd.read_csv("rfm_data.csv")
    
    df_feedback = get_feedbacks()
    print(f"👥 Tổng số phản hồi thu thập được: {len(df_feedback)}")
    
    if len(df_feedback) >= FEEDBACK_THRESHOLD and 'user_suggested_cluster' in df_feedback.columns:
        print(f"🌟 Đạt mốc {FEEDBACK_THRESHOLD} phản hồi! Chuyển đổi sang Supervised Learning (Random Forest)...")
        df_labeled = df_feedback[df_feedback['user_suggested_cluster'].notnull()].copy()
        
        if len(df_labeled) > 10:
            X_train = df_labeled[["recency", "frequency", "monetary"]]
            y_train = df_labeled["user_suggested_cluster"].astype(int)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y_train)
            
            os.makedirs("core/models", exist_ok=True)
            joblib.dump(scaler, "core/models/scaler.pkl")
            joblib.dump(rf_model, "core/models/kmeans_model.pkl") 
            
            print("✅ Đã huấn luyện thành công Random Forest Classifier!")
        else:
             print("⚠️ Phản hồi chưa đủ chất lượng để train Supervised.")
    else:
        print("🤖 Vẫn sử dụng Unsupervised Learning (K-Means). Đang train lại...")
        X = df_rfm[["Recency", "Frequency", "Monetary"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        joblib.dump(scaler, "core/models/scaler.pkl")
        joblib.dump(kmeans, "core/models/kmeans_model.pkl")
        print("✅ Đã huấn luyện lại K-Means thành công!")

    print("\n📦 Đang cập nhật lại hệ thống Gợi ý (Apriori)...")
    try:
        # Bạn có thể tuỳ chỉnh min_support/min_confidence tuỳ độ lớn data
        train_apriori(min_support=0.01, min_confidence=0.1)
    except Exception as e:
        print(f"⚠️ Lỗi khi huấn luyện Apriori: {e}")

    print("🎉 Quá trình Retraining hoàn tất.\n")

if __name__ == "__main__":
    run_retraining()
