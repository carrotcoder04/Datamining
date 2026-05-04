import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
file = "raw_input.xlsx"

df = pd.read_excel(file, sheet_name="Year 2010-2011")

print("Shape dữ liệu (chỉ lấy bảng thứ 2):", df.shape)

# =========================
# 2. CHUẨN HÓA TÊN CỘT
# =========================
df.columns = df.columns.str.strip()

# =========================
# 3. FIX CUSTOMER
# =========================
if "Customer ID" in df.columns and "CustomerID" in df.columns:
    df["CustomerID"] = df["CustomerID"].fillna(df["Customer ID"])
    df = df.drop(columns=["Customer ID"])

# =========================
# 4. MISSING
# =========================
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing (%)": missing_percent
}).sort_values(by="Missing Count", ascending=False)

print("\n=== MISSING DATA ===")
print(missing_df)

# =========================
# 5. SAVE FILE (QUAN TRỌNG)
# =========================
df.to_csv("raw_data.csv", index=False)

print("\nĐã lưu file: raw_data.csv")