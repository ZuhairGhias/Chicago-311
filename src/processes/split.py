from sklearn.model_selection import train_test_split

def split_data(df, data_cols, label_col, split_ratio):
  """
  Splits data into training and testing sets.

  Args:
    df: pandas DataFrame.
    data_cols: List of strings, names of feature columns.
    label_col: String, name of the target column.
    split_ratio: Float, the proportion of the dataset to include in the test split.

  Returns:
    X_train, X_test, y_train, y_test: Training and testing sets for features and target.
  """
  X = df[data_cols]
  y = df[label_col]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

  return X_train, X_test, y_train, y_test

# Example usage:
# Assuming 'df' is your DataFrame and you want to use 'CREATED_HOUR' and 'CREATED_DAY_OF_WEEK' to predict 'RESPONSE_TIME_DAYS' with an 80/20 split.
# data_columns = ['CREATED_HOUR', 'CREATED_DAY_OF_WEEK']
# label_column = 'RESPONSE_TIME_DAYS'
# test_size = 0.2
# X_train, X_test, y_train, y_test = split_data(df, data_columns, label_column, test_size)

# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)