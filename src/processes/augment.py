import pandas as pd
from pandas.core.arrays import string_

def add_response_time_features(df):
    """
    Calculates response time in days and categorizes it.

    Args:
        df: pandas DataFrame with 'CREATED_DATE' and 'CLOSED_DATE' columns.

    Returns:
        pandas DataFrame with 'RESPONSE_TIME', 'RESPONSE_TIME_DAYS', and 'RESPONSE_TIME_CATEGORY' columns added.
    """
    # Add a year column for convenience
    df['CREATED_YEAR'] = df['CREATED_DATE'].dt.year.astype('category')
    
    # Calculate response time
    df['RESPONSE_TIME'] = (df['CLOSED_DATE'] - df['CREATED_DATE'])

    # Calculate response time in days
    df['RESPONSE_TIME_DAYS'] = df['RESPONSE_TIME'].dt.days

    # Define a function to categorize response time
    def categorize_response_time(days):
        if days < 1:
            return '< 1 day'
        elif 1 <= days < 7:
            return '< 7 days'
        elif 7 <= days < 30:
            return '< 30 days'
        else:
            return '> 30 days'

    # Apply the function to create the response time category column
    df['RESPONSE_TIME_CATEGORY'] = df['RESPONSE_TIME_DAYS'].apply(categorize_response_time)


    return df

# Example usage of the function:
# Assuming 'df' is your DataFrame
# df = add_response_time_features(df.copy()) # Use .copy() to avoid modifying the original DataFrame if needed
# display(df[['CREATED_DATE', 'CLOSED_DATE', 'RESPONSE_TIME', 'RESPONSE_TIME_DAYS', 'RESPONSE_TIME_CATEGORY']].head())