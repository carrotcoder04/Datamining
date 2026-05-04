import pandas as pd
import pytest
from validators import validate_create_rfm_input


def test_validate_create_rfm_input_success():
    df = pd.DataFrame({
        "Invoice": ["1001", "1002"],
        "InvoiceDate": ["2021-01-01", "2021-02-01"],
        "Quantity": [1, 2],
        "UnitPrice": [10.0, 20.0],
        "CustomerID": [123, 456]
    })

    validated = validate_create_rfm_input(df)
    assert "Price" in validated.columns
    assert pd.api.types.is_datetime64_any_dtype(validated["InvoiceDate"]) or validated["InvoiceDate"].dtype == object
    assert validated["Quantity"].dtype.kind in "fi"
    assert validated["Price"].dtype.kind in "fi"


def test_validate_create_rfm_input_missing_column_raises():
    df = pd.DataFrame({
        "Invoice": ["1001"],
        "InvoiceDate": ["2021-01-01"],
        # missing Quantity
        "UnitPrice": [10.0],
        "CustomerID": [123]
    })

    with pytest.raises(ValueError):
        validate_create_rfm_input(df)
