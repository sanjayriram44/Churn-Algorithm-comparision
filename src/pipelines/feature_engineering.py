import pandas as pd
from sklearn import StandardScaler

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
        "PhoneService", "PaperlessBilling"
    ]
    df = _encode_binary_columns(df, binary_columns)

   
    df = _engineer_tenure(df)

   
    df = _encode_categorical_columns(df)

   
    df = _scale_numerical_columns(df)

   
    assert df.isnull().sum().sum() == 0, "Missing values detected"
    assert df.select_dtypes(include="object").empty, "Non-numeric columns remain"
    assert len(df) == len(y), "Feature/target length mismatch"

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
    """
    Encode binary categorical columns to numerical values.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    binary_columns (list): List of binary categorical column names.

    Returns:
    pd.DataFrame: DataFrame with encoded binary columns.
    """
    for column in binary_columns:
        df[column] = df[column].map({'Yes': 1, 'No': 0})
    return df


def _engineer_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer tenure-related features.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.

    Returns:
    pd.DataFrame: DataFrame with engineered tenure features.
    """
    df['tenure_group'] = pd.cut(df['tenure'],bins = [0, 12, 24, 48, 72], labels = ['0-12', '13-24', '25-48', '49+'])
    df.drop(columns = ['tenure'], inplace = True)
    return df

def _encode_categorical_columns(df):
    categorical_cols = [
        "gender", "Contract",
        "InternetService", "PaymentMethod",
        "tenure_group"
    ]
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def _scale_numerical_columns(df):
    scaler = StandardScaler()
    num_cols = ["MonthlyCharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df



    