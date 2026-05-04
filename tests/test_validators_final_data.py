import pandas as pd
import pytest
from validators import validate_final_data_df


def test_validate_final_data_success():
    df = pd.DataFrame({
        "CustomerID": [1,2,3],
        "Recency": [10, 20, 30],
        "Frequency": [1, 2, 3],
        "Monetary": [100.0, 200.0, 150.0]
    })

    validated = validate_final_data_df(df)
    assert "CustomerID" in validated.columns
    assert validated["Recency"].dtype.kind in "fi"


def test_validate_final_data_missing_raises():
    df = pd.DataFrame({
        "CustomerID": [1],
        # missing Recency
        "Frequency": [1],
        "Monetary": [100.0]
    })

    with pytest.raises(ValueError):
        validate_final_data_df(df)
