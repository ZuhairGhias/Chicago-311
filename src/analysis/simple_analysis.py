
import pandas as pd

def display_dataframe_info(df: pd.DataFrame):
    """
    Displays the columns and data types of a pandas DataFrame.

    Args:
        df: The pandas DataFrame to analyze.
    """
    print("Columns in the DataFrame:")
    print(df.columns)

    print("\nData types of the columns:")
    df.info()

    print("\nDescribe in more detail:")
    print(df.describe())
    
    print("\nModes for each column:")
    print(df.mode().iloc[0])
    
