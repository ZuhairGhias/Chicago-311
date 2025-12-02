import pandas as pd

def convert_types(df: pd.DataFrame):
    # convert date columns and categorical columns
    df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['CLOSED_DATE'] = pd.to_datetime(df['CLOSED_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    df['SR_TYPE'] = df['SR_TYPE'].astype('category')
    df['ORIGIN'] = df['ORIGIN'].astype('category')
    df['ZIP_CODE'] = df['ZIP_CODE'].astype('category')
    return


def clean_features(df):
    # keep only necessary columns
    cols_to_keep = ["SR_TYPE", "ORIGIN", "CREATED_DATE", "CLOSED_DATE", "ZIP_CODE", "CREATED_HOUR", "CREATED_DAY_OF_WEEK",
    "CREATED_MONTH", "LATITUDE", "LONGITUDE", "RESPONSE_TIME", "RESPONSE_TIME_DAYS", "RESPONSE_TIME_CATEGORY"]

    cols_to_drop = [c for c in df.columns if c not in cols_to_keep]

    df.drop(columns=cols_to_drop, inplace = True)

    return
