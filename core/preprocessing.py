import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Tạo thư mục chứa models
os.makedirs("core/models", exist_ok=True)
os.makedirs("backend/data", exist_ok=True)

def remove_outliers_iqr(df, columns):
    """Loại bỏ outliers dựa trên phương pháp IQR cho các cột được chỉ định"""
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    return df_out

def preprocess_and_scale(input_csv="clean_data.csv", output_scaled_csv="scaled_data.csv"):
    print("🚀 Bắt đầu quá trình Tiền xử lý (Preprocessing)...")
    
    # 1. Đọc dữ liệu
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file {input_csv}. Vui lòng chạy các bước clean_data trước.")
    
    df = pd.read_csv(input_csv)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df = df.dropna(subset=["CustomerID"])
    
    # 2. Tính toán RFM
    now = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    df["date"] = df["InvoiceDate"].dt.date
    
    # Recency
    recency_df = df.groupby("CustomerID", as_index=False)["date"].max()
    recency_df.columns = ["CustomerID", "LastPurchaseDate"]
    recency_df["Recency"] = recency_df["LastPurchaseDate"].apply(lambda x: (now.date() - x).days)
    recency_df = recency_df.drop(columns=["LastPurchaseDate"])
    
    # Frequency
    freq_df = df.drop_duplicates(subset=["Invoice", "CustomerID"])[["Invoice", "CustomerID"]]
    freq_df = freq_df.groupby("CustomerID", as_index=False)["Invoice"].count()
    freq_df.columns = ["CustomerID", "Frequency"]
    
    # Monetary
    monetary_df = df.groupby("CustomerID", as_index=False)["TotalPrice"].sum()
    monetary_df.columns = ["CustomerID", "Monetary"]
    
    rfm = recency_df.merge(freq_df, on="CustomerID").merge(monetary_df, on="CustomerID")
    print(f"📊 Kích thước RFM gốc: {rfm.shape}")
    
    # 3. Loại bỏ Outliers bằng IQR
    rfm_clean = remove_outliers_iqr(rfm, ["Recency", "Frequency", "Monetary"])
    print(f"📊 Kích thước RFM sau khi loại bỏ Outliers (IQR): {rfm_clean.shape}")
    
    # Lưu lại file rfm_data để backend dùng
    rfm_clean.to_csv("rfm_data.csv", index=False)
    
    # 4. Chuẩn hóa dữ liệu (StandardScaler)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_clean[["Recency", "Frequency", "Monetary"]])
    
    scaled_df = rfm_clean.copy()
    scaled_df[["Recency", "Frequency", "Monetary"]] = scaled_features
    
    # 5. Lưu kết quả và Model
    scaled_df.to_csv(output_scaled_csv, index=False)
    joblib.dump(scaler, "core/models/scaler.pkl")
    
    print("✅ Đã lưu file dữ liệu chuẩn hóa và scaler.pkl")
    return scaled_df

if __name__ == "__main__":
    preprocess_and_scale()
