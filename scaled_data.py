import pandas as pd
from sklearn.preprocessing import StandardScaler
from validators import validate_final_data_df

# =========================
# 1. LOAD DATA
# =========================
# This script validates the final_data.csv schema before scaling.

df = pd.read_csv("final_data.csv")

# Validate and normalize
df = validate_final_data_df(df)

# =========================
# 2. CHỌN FEATURE RFM
# =========================
features = ["Recency", "Frequency", "Monetary"]
X = df[features]

# =========================
# 3. STANDARD SCALER
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. CHUYỂN VỀ DATAFRAME
# =========================
df_scaled = pd.DataFrame(X_scaled, columns=features)

# GIỮ CustomerID
if "CustomerID" in df.columns:
    df_scaled.insert(0, "CustomerID", df["CustomerID"].values)

# =========================
# 5. KIỂM TRA
# =========================
print("Mean:")
print(df_scaled[features].mean())

print("\nStd:")
print(df_scaled[features].std())

# =========================
# 6. LƯU FILE
# =========================
df_scaled.to_csv("scaled_data.csv", index=False)

print("\nĐã lưu file: scaled_data.csv")
