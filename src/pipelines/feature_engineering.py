import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def build_features(
    df: pd.DataFrame,
    target_column: str = "Churn",
    return_target: bool = True
):
    """
    Perform feature engineering on the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    target_column : str
        Name of the target column.
    return_target : bool
        Whether to extract and return the target variable.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series (optional)
        Target vector (only if return_target=True).
    """

    df = df.copy()

 
    df = _clean_columns(df)


    if return_target:
        y = _extract_target(df, target_column)
    else:
        y = None


    df = _drop_columns(
        df,
        columns_to_drop=["customerID", "TotalCharges"]
    )


    service_columns = [
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df = _handle_service_columns(df, service_columns)


    binary_columns = [
        "Partner", "Dependents",
        "PhoneService", "PaperlessBilling",
        "SeniorCitizen"
    ]
    df = _encode_binary_columns(df, binary_columns)

   
    df = _engineer_tenure(df)

   
    df = _encode_categorical_columns(df)

  
    df = df.astype(float)

    assert df.isnull().sum().sum() == 0, "Missing values detected"

    if return_target:
        assert len(df) == len(y), "Feature/target length mismatch"
        return df, y

    return df


def _extract_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    target = df[target_column].map({"Yes": 1, "No": 0})
    df.drop(columns=[target_column], inplace=True)
    return target


def _drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    df.drop(columns=columns_to_drop, inplace=True)
    return df


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def _handle_service_columns(df: pd.DataFrame, service_columns: list) -> pd.DataFrame:
    for column in service_columns:
        df[column] = df[column].replace({"No internet service": "No"})
    return df


def _encode_binary_columns(df: pd.DataFrame, binary_columns: list) -> pd.DataFrame:
    for column in binary_columns:
        if df[column].dtype == "object":
            df[column] = df[column].map({"Yes": 1, "No": 0})
        else:
            df[column] = df[column].astype(int)
    return df


def _engineer_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12", "13-24", "25-48", "49+"]
    )
    df.drop(columns=["tenure"], inplace=True)
    return df


def _encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        df[col] = df[col].astype(str)

    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)




class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around build_features().
    """

    def __init__(self, target_column: str = "Churn"):
        self.target_column = target_column

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        return build_features(
            X,
            target_column=self.target_column,
            return_target=False
        )
