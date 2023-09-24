import pandas as pd
import numpy as np

# Read the dataset into a DataFrame
df = pd.read_csv("po2_data.csv")

# Define a function to remove rows with outliers using the IQR method for all columns
def remove_rows_with_outliers_iqr(df, threshold=1.5):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Remove rows with outliers for all columns
df_cleaned = remove_rows_with_outliers_iqr(df)
df_cleaned.shape
# Save the cleaned DataFrame to a CSV file
df_cleaned.to_csv("cleaned_data.csv", index=False)

