import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("rfm_data.csv")

print("Shape trước khi remove outlier:", df.shape)

# =========================
# 2. HÀM REMOVE OUTLIER (IQR)
# =========================
def remove_outliers_iqr(data, cols):
    df_clean = data.copy()

    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"\n--- {col} ---")
        print("Lower bound:", lower_bound)
        print("Upper bound:", upper_bound)

        before = df_clean.shape[0]

        # 🔥 Chỉ filter theo cột số, KHÔNG đụng CustomerID
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) &
            (df_clean[col] <= upper_bound)
        ]

        after = df_clean.shape[0]
        print(f"Removed {before - after} outliers")

    return df_clean


# =========================
# 3. ÁP DỤNG
# =========================
cols = ["Recency", "Frequency", "Monetary"]

df_no_outlier = remove_outliers_iqr(df, cols)

# =========================
# 4. KẾT QUẢ
# =========================
print("\nShape sau khi remove outlier:", df_no_outlier.shape)

# =========================
# 5. GIỮ CustomerID (đảm bảo)
# =========================
# (Thực ra vẫn giữ, nhưng check cho chắc)
required_cols = ["CustomerID", "Recency", "Frequency", "Monetary"]
df_no_outlier = df_no_outlier[required_cols]

# =========================
# 6. LƯU FILE
# =========================
df_no_outlier.to_csv("final_data.csv", index=False)

print("\n✅ Đã lưu file: final_data.csv")