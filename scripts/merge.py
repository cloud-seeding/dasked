import os
import pandas as pd

def rename_columns(df):
    def transform_column_name(col_name):
        if "_level_" in col_name:
            parts = col_name.split("_level_")
            if len(parts) == 2 and parts[1].isdigit():
                var, n = parts
                return f"{float(n)}-{var}"
        return col_name

    # Apply the transformation to all column names
    df.columns = [transform_column_name(col) for col in df.columns]
    return df

# Define the folder containing the CSV files
folder_path = "T-2"
output_file = os.path.join(folder_path, "T-2.csv")

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to hold the DataFrames
dataframes = []

# Loop through each CSV file and read it
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dataframes, ignore_index=True)

merged_df = rename_columns(merged_df)

# Write the concatenated DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved to: {output_file}")