import matplotlib.pyplot as plt
import seaborn as sns

def plot_response_time_categories(df):
    """
    Plots the distribution of response time categories.

    Args:
        df: pandas DataFrame with a 'RESPONSE_TIME_CATEGORY' column.
        title: String, the title of the plot.
    """
    title="Distribution of Response Time Categories"

    if 'RESPONSE_TIME_CATEGORY' not in df.columns:
        print("Error: 'RESPONSE_TIME_CATEGORY' column not found in the DataFrame.")
        return

    # Count the occurrences of each response time category
    response_time_counts = df['RESPONSE_TIME_CATEGORY'].value_counts().sort_index()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=response_time_counts.index, y=response_time_counts.values, palette='viridis')
    plt.title(title)
    plt.xlabel('Response Time Category')
    plt.ylabel('Number of Service Requests')
    plt.show()

def plot_average_response_time_by_column(df, grouping_column):
    """
    Plots the average response time in days grouped by a specified column.

    Args:
        df: pandas DataFrame with the grouping_column and 'RESPONSE_TIME_DAYS' columns.
        grouping_column: String, the name of the column to group by.
    """
    if grouping_column not in df.columns or 'RESPONSE_TIME_DAYS' not in df.columns:
        print(f"Error: '{grouping_column}' or 'RESPONSE_TIME_DAYS' column not found in the DataFrame.")
        return

    # Calculate the average response time by the grouping column
    average_response_time_by_group = df.groupby(grouping_column, observed=False)['RESPONSE_TIME_DAYS'].mean().sort_values(ascending=False)

    # Select the top N categories if there are too many for a clear plot
    if len(average_response_time_by_group) > 20: # Arbitrary threshold, can be adjusted
        print(f"Warning: More than 20 unique values in '{grouping_column}'. Plotting top 20.")
        average_response_time_by_group = average_response_time_by_group.head(20)

    # Create a horizontal bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=average_response_time_by_group.values, y=average_response_time_by_group.index, hue=average_response_time_by_group.index, palette='viridis', legend=False)
    plt.title(f'Average Response Time by {grouping_column}') # Dynamic title
    plt.xlabel('Average Response Time (Days)')
    plt.ylabel(grouping_column)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'df' is your DataFrame
# plot_average_response_time_by_column(df, 'SR_TYPE')
# plot_average_response_time_by_column(df, 'ORIGIN')
# plot_average_response_time_by_column(df, 'CREATED_MONTH')