import pandas as pd

# Normalizes column names and validates required columns for pipeline steps.
# Functions raise ValueError with a clear message when validation fails.

_ALIAS_MAP = {
    # common alternative names mapping to canonical names used by the pipeline
    "UnitPrice": "Price",
    "Unit Price": "Price",
    "Unit_Price": "Price",
    "InvoiceNo": "Invoice",
    "InvoiceNo ": "Invoice",
    "Invoice No": "Invoice",
    "Customer ID": "CustomerID",
    "Customer_Id": "CustomerID",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    new_cols = []
    for c in cols:
        if c in _ALIAS_MAP:
            new_cols.append(_ALIAS_MAP[c])
        else:
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df


def validate_create_rfm_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a DataFrame intended for Create_RFM.py.

    Ensures presence of: Invoice (or alias), InvoiceDate, Quantity, Price/UnitPrice, CustomerID.
    Returns a DataFrame with normalized column names (Price, Invoice, CustomerID) suitable for downstream steps.
    Raises ValueError with actionable message on failure.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = _normalize_columns(df)

    required = ["Invoice", "InvoiceDate", "Quantity", "Price", "CustomerID"]
    missing = [c for c in required if c not in df.columns]

    # As a last attempt, accept 'UnitPrice' alias already normalized above.
    if missing:
        raise ValueError(f"Missing required columns for RFM creation: {missing}. "
                         "Expected columns (aliases allowed): Invoice/InvoiceNo, InvoiceDate, Quantity, Price/UnitPrice, CustomerID")

    # Coerce types where appropriate and provide informative errors
    # InvoiceDate -> datetime
    try:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    except Exception:
        raise ValueError("Cannot parse InvoiceDate column to datetime. Ensure values are ISO/parsable dates.")

    if df["InvoiceDate"].isna().all():
        raise ValueError("InvoiceDate column parsed to all NaT; check input values and format.")

    # Quantity and Price numeric
    for col in ("Quantity", "Price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            raise ValueError(f"Column {col} could not be converted to numeric. Check values.")

    # CustomerID presence (non-null)
    if df["CustomerID"].isnull().all():
        raise ValueError("CustomerID column is missing or all null. CustomerID is required.")

    return df


def validate_final_data_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate final_data.csv used by scaled_data.py.

    Ensures presence of Recency, Frequency, Monetary and CustomerID; enforces numeric types for R/F/M.
    Returns normalized DataFrame or raises ValueError.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    df = _normalize_columns(df)

    required = ["CustomerID", "Recency", "Frequency", "Monetary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in final_data: {missing}. Expected CustomerID, Recency, Frequency, Monetary.")

    # Ensure numeric
    for col in ("Recency", "Frequency", "Monetary"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            raise ValueError(f"Column {col} could not be converted to numeric in final_data.csv")

    return df
