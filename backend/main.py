import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from backend.database import init_db, insert_feedback, insert_retrain_log
from core.recommender import get_recommendations, get_apriori_recommendations
from mlops.retrain import check_and_trigger_feedback_retrain

# Khởi tạo DB
init_db()

app = FastAPI(title="Datamining MLOps API")

# Cấu hình CORS cho Frontend gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_models():
    try:
        scaler = joblib.load("core/models/scaler.pkl")
        kmeans_model = joblib.load("core/models/kmeans_model.pkl")
        return scaler, kmeans_model
    except Exception as e:
        print(f"⚠️ Lỗi khi load models: {e}")
        return None, None

class PredictRequest(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float

class FeedbackRequest(BaseModel):
    customer_id: str
    recency: float
    frequency: float
    monetary: float
    predicted_cluster: int
    correct: bool
    user_suggested_cluster: Optional[int] = None

class BasketRequest(BaseModel):
    products: List[str]  # Danh sách sản phẩm người quản lý chọn
    top_n: Optional[int] = 5

@app.get("/")
def read_root():
    return {"message": "Datamining MLOps API is running!"}

@app.post("/predict")
def predict_cluster(req: PredictRequest):
    scaler, kmeans_model = load_models()
    if not scaler or not kmeans_model:
        return {"error": "Models chưa sẵn sàng."}
    input_data = np.array([[req.Recency, req.Frequency, req.Monetary]])
    scaled_data = scaler.transform(input_data)
    cluster = kmeans_model.predict(scaled_data)[0]
    return {
        "Recency": req.Recency,
        "Frequency": req.Frequency,
        "Monetary": req.Monetary,
        "Predicted_Cluster": int(cluster)
    }

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    try:
        insert_feedback(
            req.customer_id,
            req.recency, req.frequency, req.monetary,
            req.predicted_cluster, req.correct, req.user_suggested_cluster
        )
        # ── Kích hoạt Retrain ngay nếu đủ feedback SAI ───────────────────────
        if not req.correct:
            import threading
            t = threading.Thread(target=check_and_trigger_feedback_retrain, daemon=True)
            t.start()
        return {"message": "Cảm ơn phản hồi của bạn! Hệ thống sẽ tự học từ dữ liệu này."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("backend/data", exist_ok=True)
    file_location = f"backend/data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"message": f"File '{file.filename}' đã được upload thành công.", "path": file_location}

# --- MODULE RECOMMENDATION ENGINE ---
@app.get("/recommend/{customer_id}")
def recommend_products_cluster(customer_id: str):
    try:
        cid = float(customer_id)
    except:
        cid = customer_id
    result = get_recommendations(cid, top_n=5)
    return result

@app.get("/recommend_apriori")
def recommend_products_apriori(product: str):
    result = get_apriori_recommendations(product, top_n=5)
    return result


@app.post("/recommend_apriori_basket")
def recommend_basket(req: BasketRequest):
    """
    Nhận vào một GIỎ HÀNG gồm nhiều sản phẩm (n >= 1).
    Trả về danh sách sản phẩm gợi ý mua kèm (dựa trên luật kết hợp FP-Growth),
    được tổng hợp và xếp hạng theo tần suất xuất hiện trong các luật.
    """
    if not req.products:
        return {"error": "Danh sách sản phẩm không được rỗng."}

    import joblib
    RULES_PATH = "core/models/association_rules.pkl"
    if not os.path.exists(RULES_PATH):
        return {"error": "Chưa có mô hình FP-Growth. Vui lòng chạy train_apriori() trước."}

    rules = joblib.load(RULES_PATH)
    product_scores: dict = {}  # product_name -> score

    for product in req.products:
        product_upper = product.strip().upper()
        # FIX: So sánh sau khi strip cả 2 phía → tránh lỗi trailing space
        matching = rules[rules["antecedents"].apply(
            lambda s: any(x.strip().upper() == product_upper for x in s)
        )]
        input_uppers = [p.strip().upper() for p in req.products]
        for _, row in matching.iterrows():
            for item in row["consequents"]:
                item_clean = item.strip()
                if item_clean.upper() not in input_uppers:  # Không gợi ý lại SP đã chọn
                    product_scores[item_clean] = product_scores.get(item_clean, 0) + row["lift"]

    if not product_scores:
        return {
            "input_products": req.products,
            "recommendations": [],
            "note": "Không tìm thấy luật kết hợp nào cho giỏ hàng này."
        }

    # Sắp xếp theo tổng lift giảm dần
    sorted_items = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [
        {"rank": i + 1, "product": prod, "lift_score": round(score, 4)}
        for i, (prod, score) in enumerate(sorted_items[: req.top_n])
    ]

    return {
        "input_products": req.products,
        "total_rules_matched": len(product_scores),
        "recommendations": recommendations
    }

# --- DEBUG ENDPOINT ---
@app.get("/debug/customer/{customer_id}")
def debug_customer(customer_id: str):
    """
    Debug: Kiểm tra CustomerID tồn tại ở đâu.
    Truy cập trực tiếp: http://localhost:8000/debug/customer/17850
    """
    result = {"customer_id_input": customer_id}
    try:
        cid_float = float(customer_id)
        result["converted_to_float"] = cid_float
    except Exception as e:
        result["conversion_error"] = str(e)
        return result

    if os.path.exists("rfm_data_with_clusters.csv"):
        df = pd.read_csv("rfm_data_with_clusters.csv")
        result["rfm_total_customers"] = len(df)
        result["rfm_customerid_dtype"] = str(df["CustomerID"].dtype)
        result["rfm_sample_ids"] = df["CustomerID"].head(5).tolist()
        found = df[df["CustomerID"] == cid_float]
        result["found_in_rfm"] = not found.empty
        if found.empty:
            result["note"] = (
                "KH này bị loại khỏi RFM do lọc Outlier (IQR). "
                "Vui lòng dùng một ID khác từ danh sách rfm_sample_ids."
            )
        else:
            result["cluster"] = int(found.iloc[0]["Cluster"])
    else:
        result["rfm_file_status"] = "MISSING"

    return result
