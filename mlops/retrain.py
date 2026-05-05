import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from core.recommender import train_apriori
from backend.database import insert_retrain_log

DB_PATH = "backend/feedback.db"
FEEDBACK_THRESHOLD = 50   # Ngưỡng feedback để switch sang Supervised
# ──── Ngưỡng feedback SAI để kích hoạt retrain ngay lập tức ────────────────
WRONG_FEEDBACK_TRIGGER = 10  # Cứ 10 feedback báo sai → retrain

def get_feedbacks():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feedbacks", conn)
    conn.close()
    return df

def count_wrong_feedbacks():
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedbacks WHERE correct = 0")
    count = cur.fetchone()[0]
    conn.close()
    return count

def _train_model(df_rfm: pd.DataFrame, df_feedback: pd.DataFrame, trigger: str):
    """
    Hàm nội bộ thực hiện huấn luyện mô hình.
    Tự động chọn giữa KMeans và RandomForest dựa trên số lượng feedback.
    """
    os.makedirs("core/models", exist_ok=True)
    df_labeled = pd.DataFrame()
    if not df_feedback.empty and "user_suggested_cluster" in df_feedback.columns:
        df_labeled = df_feedback[df_feedback["user_suggested_cluster"].notnull()].copy()

    if len(df_labeled) >= FEEDBACK_THRESHOLD:
        print(f"🌟 [{trigger}] Đạt mốc {FEEDBACK_THRESHOLD} feedback! Chuyển sang Supervised Learning (Random Forest)...")
        X_train = df_labeled[["recency", "frequency", "monetary"]]
        y_train = df_labeled["user_suggested_cluster"].astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_train)
        joblib.dump(scaler, "core/models/scaler.pkl")
        joblib.dump(model, "core/models/kmeans_model.pkl")
        note = f"Random Forest trained with {len(df_labeled)} labeled feedbacks."
        print(f"✅ [{trigger}] Random Forest Classifier đã được huấn luyện!")
    else:
        print(f"🤖 [{trigger}] Unsupervised (K-Means). Đang train lại trên {len(df_rfm)} khách hàng...")
        X = df_rfm[["Recency", "Frequency", "Monetary"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        joblib.dump(scaler, "core/models/scaler.pkl")
        joblib.dump(kmeans, "core/models/kmeans_model.pkl")
        note = f"K-Means retrained on {len(df_rfm)} customers. Wrong feedbacks: {len(df_labeled)}."
        print(f"✅ [{trigger}] K-Means đã được huấn luyện lại thành công!")

    print(f"\n📦 [{trigger}] Đang cập nhật hệ thống Gợi ý (FP-Growth/Apriori)...")
    try:
        train_apriori(min_support=0.01, min_confidence=0.1)
    except Exception as e:
        print(f"⚠️ Lỗi khi huấn luyện Apriori: {e}")

    insert_retrain_log(trigger=trigger, status="SUCCESS", note=note)
    print(f"🎉 [{trigger}] Retraining hoàn tất.\n")


def run_retraining(trigger: str = "scheduled_2am"):
    """
    Chạy quá trình retrain.
    
    trigger options:
      - "scheduled_2am"   : Lịch trình tự động mỗi đêm
      - "feedback_trigger": Kích hoạt vì có >= WRONG_FEEDBACK_TRIGGER feedback sai
      - "data_update"     : Dữ liệu mới được upload, cần train lại ngay
    """
    print("\n" + "="*60)
    print(f"🔄 BẮT ĐẦU RETRAINING — Trigger: [{trigger.upper()}]")
    print("="*60)

    if not os.path.exists("rfm_data.csv"):
        msg = "⚠️ Không có rfm_data.csv. Hủy retrain."
        print(msg)
        insert_retrain_log(trigger=trigger, status="SKIPPED", note=msg)
        return

    df_rfm = pd.read_csv("rfm_data.csv")
    df_feedback = get_feedbacks()
    print(f"👥 Tổng feedback: {len(df_feedback)} | Feedback sai cụm: {count_wrong_feedbacks()}")

    _train_model(df_rfm, df_feedback, trigger)


def check_and_trigger_feedback_retrain():
    """
    Kiểm tra số lượng feedback sai. Nếu vượt ngưỡng → retrain ngay lập tức.
    Hàm này được gọi mỗi khi API /feedback nhận được phản hồi mới.
    """
    wrong_count = count_wrong_feedbacks()
    print(f"🔍 Kiểm tra feedback trigger: {wrong_count}/{WRONG_FEEDBACK_TRIGGER} feedback sai.")
    if wrong_count > 0 and wrong_count % WRONG_FEEDBACK_TRIGGER == 0:
        print(f"🚨 Đạt ngưỡng {WRONG_FEEDBACK_TRIGGER} feedback sai → Kích hoạt Retrain ngay!")
        run_retraining(trigger="feedback_trigger")


if __name__ == "__main__":
    run_retraining(trigger="manual")
