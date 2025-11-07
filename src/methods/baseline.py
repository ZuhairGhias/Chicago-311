import pandas as pd
import numpy as np

def _get_single_prior_year_average(row, historical_data, grouping_cols):
    """
    Calculates the average response time from the immediately preceding year for a single row.
    Helper function for get_all_prior_year_average.
    """
    current_created_date = row['CREATED_DATE']

    # Filter historical data for the one-year window immediately preceding the current created date
    prior_year_start_date = current_created_date - pd.DateOffset(years=1)
    
    prior_year_data = historical_data[
        (historical_data['CREATED_DATE'] >= prior_year_start_date) &
        (historical_data['CREATED_DATE'] < current_created_date)
    ].copy()

    if prior_year_data.empty:
        return np.nan

    grouping_values = tuple(row[col] for col in grouping_cols)

    if grouping_cols:
        filter_condition = pd.Series(True, index=prior_year_data.index)
        for i, col in enumerate(grouping_cols):
            # Assuming grouping_cols are present in historical_data as per the problem context
            filter_condition &= (prior_year_data[col] == grouping_values[i])

        filtered_prior_data = prior_year_data[filter_condition]

        if not filtered_prior_data.empty:
            return filtered_prior_data['RESPONSE_TIME_DAYS'].mean()
        else:
            return np.nan  # No matching prior year data for this group
    else:
        # If no grouping columns, calculate overall average from prior year data within the window
        return prior_year_data['RESPONSE_TIME_DAYS'].mean()

def get_all_prior_year_average(current_data, historical_data, grouping_cols):
    """
    Calculates the average response time from the immediately preceding year for each entry in current_data.
    The average is based on the specified grouping columns.

    Args:
        current_data (pd.DataFrame): DataFrame containing the current observations for which to make predictions.
        historical_data (pd.DataFrame): DataFrame containing all historical data.
        grouping_cols (list): List of column names to group by for calculating the average.

    Returns:
        pd.Series: A Series containing the predicted average response times for each row in current_data.
    """
    predictions = []
    for _, row in current_data.iterrows():
        predictions.append(_get_single_prior_year_average(row, historical_data, grouping_cols))
    return pd.Series(predictions, index=current_data.index)