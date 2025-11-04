import pandas as pd

def add_response_time_features(df):
    """
    Calculates response time in days and categorizes it.

    Args:
        df: pandas DataFrame with 'CREATED_DATE' and 'CLOSED_DATE' columns.

    Returns:
        pandas DataFrame with 'RESPONSE_TIME', 'RESPONSE_TIME_DAYS', and 'RESPONSE_TIME_CATEGORY' columns added.
    """
    # Calculate response time
    df['RESPONSE_TIME'] = (df['CLOSED_DATE'] - df['CREATED_DATE'])

    # Calculate response time in days
    df['RESPONSE_TIME_DAYS'] = df['RESPONSE_TIME'].dt.days

    # Define categories and labels
    bins = [-1, 1, 7, 30, float('inf')]
    labels = ['< 1 day', '< 7 days', '< 30 days', '> 30 days']

    # Create the response time category column
    df['RESPONSE_TIME_CATEGORY'] = pd.cut(df['RESPONSE_TIME_DAYS'], bins=bins, labels=labels, right=True)

    return df

# Example usage of the function:
# Assuming 'df' is your DataFrame
# df = add_response_time_features(df.copy()) # Use .copy() to avoid modifying the original DataFrame if needed
# display(df[['CREATED_DATE', 'CLOSED_DATE', 'RESPONSE_TIME', 'RESPONSE_TIME_DAYS', 'RESPONSE_TIME_CATEGORY']].head())