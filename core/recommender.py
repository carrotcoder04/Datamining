import pandas as pd
import numpy as np
import os
import joblib
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths
CLEAN_DATA_PATH = "clean_data.csv" 
CLUSTER_DATA_PATH = "rfm_data_with_clusters.csv"
RULES_PATH = "core/models/association_rules.pkl"

def get_recommendations(customer_id, top_n=5):
    """
    Gợi ý sản phẩm dựa trên Collaborative Filtering via Clustering.
    Nếu KH bị lọc khỏi RFM (outlier), tự tính lại và predict cụm.
    """
    if not os.path.exists(CLEAN_DATA_PATH) or not os.path.exists(CLUSTER_DATA_PATH):
        return {"error": "Chưa có đủ dữ liệu để gợi ý. Vui lòng kiểm tra lại clean_data.csv và rfm_data_with_clusters.csv"}

    # 1. Tìm cụm của Customer (chuẩn hóa kiểu dữ liệu trước khi so sánh)
    df_cluster = pd.read_csv(CLUSTER_DATA_PATH)
    # Chuẩn hóa: thử float trước (vì CSV lưu CustomerID dạng 17850.0)
    try:
        customer_id_lookup = float(customer_id)
    except (ValueError, TypeError):
        customer_id_lookup = customer_id
    customer_info = df_cluster[df_cluster['CustomerID'] == customer_id_lookup]
    is_outlier_fallback = False

    if customer_info.empty:
        # Fallback: KH bị lọc do IQR → tự tính RFM và predict cụm bằng model
        df_tx = pd.read_csv(CLEAN_DATA_PATH)
        df_tx['CustomerID'] = pd.to_numeric(df_tx['CustomerID'], errors='coerce')
        df_tx['InvoiceDate'] = pd.to_datetime(df_tx['InvoiceDate'])
        df_tx['TotalPrice'] = df_tx['Quantity'] * df_tx['Price']
        cust_tx = df_tx[df_tx['CustomerID'] == customer_id_lookup]
        if cust_tx.empty:
            return {"error": f"CustomerID {customer_id} không tồn tại trong dữ liệu giao dịch."}
        # Tính RFM
        now = df_tx['InvoiceDate'].max() + pd.Timedelta(days=1)
        recency = (now - cust_tx['InvoiceDate'].max()).days
        frequency = cust_tx['Invoice'].nunique()
        monetary = cust_tx['TotalPrice'].sum()
        # Predict cụm
        scaler_path, model_path = "core/models/scaler.pkl", "core/models/kmeans_model.pkl"
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            return {"error": "Chưa có model. Vui lòng chạy core/clustering.py trước."}
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        cluster_label = int(model.predict(scaler.transform(np.array([[recency, frequency, monetary]])))[0])
        is_outlier_fallback = True
    else:
        cluster_label = int(customer_info.iloc[0]['Cluster'])
    
    # Lấy danh sách KH cùng cụm
    customers_in_cluster = df_cluster[df_cluster['Cluster'] == cluster_label]['CustomerID'].tolist()
    
    # 2. Đọc dữ liệu giao dịch
    df_transactions = pd.read_csv(CLEAN_DATA_PATH)
    df_transactions['CustomerID'] = pd.to_numeric(df_transactions['CustomerID'], errors='coerce')
    
    # 3. Xây dựng Ma trận mua hàng (Customer x Product) cho toàn cụm
    # Chỉ lấy KH trong cụm + KH hiện tại
    all_customer_ids = customers_in_cluster + [customer_id_lookup]
    df_cluster_tx = df_transactions[df_transactions['CustomerID'].isin(all_customer_ids)]
    
    # Tạo ma trận: mỗi hàng là 1 KH, mỗi cột là 1 sản phẩm
    # Giá trị = 1 nếu KH đã mua, 0 nếu chưa (Binary purchase matrix)
    purchase_matrix = (
        df_cluster_tx.groupby(['CustomerID', 'Description'])['Quantity']
        .sum()
        .unstack(fill_value=0)
        .clip(upper=1)  # Binary: đã mua (1) hay chưa (0)
    )
    
    if customer_id_lookup not in purchase_matrix.index:
        return {"error": f"Không tìm thấy lịch sử mua hàng của CustomerID {customer_id}."}
    
    # 4. Tính Cosine Similarity giữa KH hiện tại với tất cả KH trong cụm
    from sklearn.metrics.pairwise import cosine_similarity
    target_vector = purchase_matrix.loc[[customer_id_lookup]].values
    cluster_matrix = purchase_matrix.drop(index=customer_id_lookup, errors='ignore').values
    cluster_ids = purchase_matrix.drop(index=customer_id_lookup, errors='ignore').index.tolist()
    
    similarities = cosine_similarity(target_vector, cluster_matrix)[0]
    
    # 5. Lấy Top-K KH giống nhất (k=20)
    K = min(20, len(similarities))
    top_k_indices = similarities.argsort()[::-1][:K]
    similar_customers = [cluster_ids[i] for i in top_k_indices if similarities[i] > 0]
    
    if not similar_customers:
        # Fallback: không tìm được KH tương đồng, dùng top phổ biến trong cụm
        similar_customers = cluster_ids[:20]
    
    # 6. Tìm sản phẩm mà KH tương đồng đã mua nhưng KH hiện tại chưa mua
    customer_bought = set(purchase_matrix.columns[purchase_matrix.loc[customer_id_lookup] > 0].tolist())
    
    # Tính điểm số cho từng sản phẩm: tổng số KH tương đồng đã mua (có tính trọng số theo độ tương đồng)
    product_scores = {}
    for idx, sim_cid in enumerate(similar_customers):
        sim_score = similarities[top_k_indices[idx]] if idx < len(top_k_indices) else 0
        if sim_cid not in purchase_matrix.index:
            continue
        sim_cid_bought = set(purchase_matrix.columns[purchase_matrix.loc[sim_cid] > 0].tolist())
        new_products = sim_cid_bought - customer_bought
        for prod in new_products:
            product_scores[prod] = product_scores.get(prod, 0) + sim_score
    
    # Sắp xếp theo điểm số giảm dần
    sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return {
        "customer_id": customer_id,
        "cluster": cluster_label,
        "cluster_size": len(customers_in_cluster),
        "similar_customers_used": len(similar_customers),
        "is_outlier_customer": is_outlier_fallback,
        "note": "KH này là outlier. Cụm được dự đoán bằng model." if is_outlier_fallback else None,
        "recommendations": [
            {"rank": i+1, "product": prod, "similarity_score": round(score, 4)}
            for i, (prod, score) in enumerate(sorted_products)
        ]
    }

def train_apriori(min_support=0.01, min_confidence=0.1):
    """
    Huấn luyện luật kết hợp Apriori và lưu model
    """
    print("🚀 Bắt đầu huấn luyện Apriori (Association Rules)...")
    if not os.path.exists(CLEAN_DATA_PATH):
        print("⚠️ Không tìm thấy clean_data.csv")
        return
        
    df = pd.read_csv(CLEAN_DATA_PATH)
    # Lấy ngẫu nhiên mẫu nhỏ nếu data quá lớn để tránh tràn RAM (Tùy chọn)
    # df = df.sample(frac=0.3, random_state=42)
    
    print("🧹 Chuẩn bị giỏ hàng...")
    # Xử lý dữ liệu giỏ hàng: mỗi InvoiceNo là 1 giỏ
    basket = (df.groupby(['Invoice', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('Invoice'))
    
    # Chuyển số lượng > 0 thành True, <= 0 thành False (Tối ưu RAM gấp 8 lần so với int64)
    def encode_units(x):
        return bool(x >= 1)
    
    basket_sets = basket.map(encode_units).astype(bool)
    
    print("⏳ Đang chạy thuật toán FP-Growth với min_support=0.5%...")
    # ── FIX: Strip khoảng trắng thừa trong tên sản phẩm trước khi build basket
    basket_sets.columns = [c.strip() for c in basket_sets.columns]
    frequent_itemsets = fpgrowth(basket_sets, min_support=0.005, use_colnames=True)
    # Chuẩn hóa: strip tất cả tên sản phẩm trong itemsets
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
        lambda s: frozenset(x.strip() for x in s)
    )
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sắp xếp rules theo lift và confidence
    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    # Lưu lại
    os.makedirs("core/models", exist_ok=True)
    joblib.dump(rules, RULES_PATH)
    print("✅ Đã lưu tập luật Apriori thành công!")

def get_apriori_recommendations(product_name, top_n=5):
    """
    Lấy gợi ý sản phẩm mua kèm dựa trên Apriori.
    Tự động xử lý khoảng trắng thừa (trailing/leading space) để tìm chính xác hơn.
    """
    if not os.path.exists(RULES_PATH):
        return {"error": "Chưa có mô hình Apriori. Vui lòng đợi hệ thống huấn luyện ban đêm hoặc chạy thủ công train_apriori()."}
        
    rules = joblib.load(RULES_PATH)
    
    # FIX: So sánh sau khi strip cả 2 phía → khắc phục lỗi trailing space
    product_clean = product_name.strip().upper()

    def check_item_in_antecedents(antecedents_set):
        return any(x.strip().upper() == product_clean for x in antecedents_set)
        
    matching_rules = rules[rules['antecedents'].apply(check_item_in_antecedents)]
    
    if matching_rules.empty:
        return {
            "product": product_name,
            "recommendations": [],
            "note": f"Không tìm thấy luật kết hợp nào cho '{product_name}'. "
                    "Sản phẩm có thể chưa đủ phổ biến (< 0.5% đơn hàng)."
        }
        
    # Sắp xếp theo lift giảm dần để lấy gợi ý tốt nhất trước
    matching_rules = matching_rules.sort_values("lift", ascending=False)

    # Lấy các mặt hàng ở vế phải (consequents), đã strip
    flat_list = []
    for consequents in matching_rules['consequents']:
        for item in consequents:
            item_clean = item.strip()
            if item_clean not in flat_list:
                flat_list.append(item_clean)
                
    return {
        "product": product_name,
        "total_rules_matched": len(matching_rules),
        "recommendations": flat_list[:top_n]
    }

if __name__ == "__main__":
    pass
