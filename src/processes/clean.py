import pandas as pd

def convert_types(df: pd.DataFrame):
    """
    Displays the columns and data types of a pandas DataFrame.

    Args:
        df: The pandas DataFrame to analyze.
    """
    # Convert date columns to datetime objects
    df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce').dt.date
    df['CLOSED_DATE'] = pd.to_datetime(df['CLOSED_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce').dt.date

    # Convert 'ZIP_CODE' to integer, handling potential NaNs by allowing nullable integer type
    # or converting NaNs first if needed. Using errors='coerce' for conversion.
    # A nullable integer type handles NaNs directly.
    df['SR_TYPE'] = df['SR_TYPE'].astype('category')
    df['ORIGIN'] = df['ORIGIN'].astype('category')
    df['ZIP_CODE'] = df['ZIP_CODE'].astype('category')
    return


def clean_features(df):
    """Return a copy of ``df`` with duplicates dropped and optional columns removed."""

    cols_to_keep = ["SR_TYPE", "ORIGIN", "CREATED_DATE", "CLOSED_DATE", "ZIP_CODE", "CREATED_HOUR", "CREATED_DAY_OF_WEEK", 
    "CREATED_MONTH", "LATITUDE", "LONGITUDE"]

    cols_to_drop = [c for c in df.columns if c not in cols_to_keep]

    df.drop(columns=cols_to_drop, inplace = True)

    return
  
