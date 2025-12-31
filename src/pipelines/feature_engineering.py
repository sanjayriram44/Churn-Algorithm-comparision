import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_features(df: pd.DataFrame, target_column: str = "Churn"):
    """
    Perform feature engineering on the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    target_column (str): Name of the target column.

    Returns:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target vector
    """


    df = df.copy()

   
    df = _clean_columns(df)

    
    y = _extract_target(df, target_column)

   
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
        "PhoneService", "PaperlessBilling",  "SeniorCitizen" 
    ]
    df = _encode_binary_columns(df, binary_columns)

   
    df = _engineer_tenure(df)

   
    df = _encode_categorical_columns(df)

   
    df = _scale_numerical_columns(df)


    non_numeric_cols = df.select_dtypes(include="object").columns.tolist()

    if non_numeric_cols:
        print(" Non-numeric columns remaining after feature engineering:")
        for col in non_numeric_cols:
            print(f" - {col}: {df[col].unique()[:5]}")
        raise AssertionError("Non-numeric columns remain")


    na_cols = df.columns[df.isnull().any()]
    if len(na_cols) > 0:
        print(" Columns with missing values:")
        for col in na_cols:
            print(f"{col}: {df[col].isnull().sum()}")
    
    
    df = df.astype(float)

    assert df.isnull().sum().sum() == 0, "Missing values detected"
    assert len(df) == len(y), "Feature/target length mismatch"
    print(df.dtypes[df.dtypes == "object"])

    return df, y



def _extract_target(df, target_column: str) -> pd.Series:
    """
    Extract the target variable from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
    pd.Series: Series containing the target variable.
    """
    target = df[f"{target_column}"].map({"Yes": 1, "No": 0})
    df.drop(columns=[f"{target_column}"], inplace=True)
    return target

def _drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    columns_to_drop (list): List of column names to drop.

    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def _clean_columns(df: pd.DataFrame):
    """
    Clean column names by stripping whitespace and converting to lowercase.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
    pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.strip()
    return df

def _handle_service_columns(df: pd.DataFrame, service_columns: list) -> pd.DataFrame:
    """
    Handle service-related columns by converting 'No internet service' to 'No'.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    service_columns (list): List of service-related column names.

    Returns:
    pd.DataFrame: DataFrame with updated service-related columns.
    """
    for column in service_columns:
        df[column] = df[column].replace({'No internet service': 'No'})
    return df

def _encode_binary_columns(df: pd.DataFrame, binary_columns: list) -> pd.DataFrame:
    
    for column in binary_columns:
        if df[column].dtype == "object":
            df[column] = df[column].map({'Yes': 1, 'No': 0})
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


def _scale_numerical_columns(df):
    scaler = StandardScaler()
    num_cols = ["MonthlyCharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df



    