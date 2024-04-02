import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import os
import glob

import os
import glob
import pandas as pd
import numpy as np

# Get the current working directory
current_directory = os.getcwd()

# Find files ending with ".col" in the current directory
col_files = glob.glob(os.path.join(current_directory, "*.col"))

def remove_values_within_threshold(df, column_name, threshold):
    column_values = df[column_name]

    # Iterate over each value in the column
    for value in column_values:
        # Determine the range where the value belongs
        range_start = np.floor(value)
        range_end = np.ceil(value)

        # Count the number of values within the range
        count = ((column_values >= range_start) & (column_values < range_end)).sum()

        # If the count is less than or equal to the threshold, remove the value
        if count <= threshold:
            df.loc[df[column_name] == value, column_name] = 'sagar'



# Define function to count values between whole numbers with step of 1
def count_values_between(column_values):
    total_count = len(column_values)
    min_value = np.floor(column_values.min())
    max_value = np.ceil(column_values.max())
    
    counts = []
    for i in range(int(min_value), int(max_value)):
        count = ((column_values >= i) & (column_values < i + 1)).sum()
        if count != 0:  # Only include if count is non-zero
            percentages = (count / total_count) * 100 
            counts.append((count, f"'{i}<{i+1} = {count} ({percentages:.2f}%)'"))
    
    return counts

# Count values between whole numbers with step of 1 for each specified column
columns_to_count = ["NS(cm)", "EW(cm)", "UD(cm)"]  # Specify columns here
results = {}
max_length = 0  # Track the maximum length of counts



# Iterate through each .col file found
for file_path in col_files:


    if (file_path == "OPUS.col"):
        delimiter= '\t'
    else:
        delimiter= '\s+'
    # Load the data into a pandas DataFrame
    # Load the data into a pandas DataFrame
    df = pd.read_csv(file_path, sep=delimiter, header=1, dtype={"NS(cm)": float, "EW(cm)": float, "UD(cm)": float})

    # Select the first four columns using iloc
    df = df.iloc[:, :4]
    column_names = ["Decimal-Year", "NS(cm)", "EW(cm)", "UD(cm)"]
    df.columns = column_names


    # Count values between whole numbers for each specified column
    for column in df.columns[1:]:
        if column not in results:
            results[column] = []
        counts = count_values_between(df[column])
        counts.sort(reverse=True)  # Sort counts based on count value
        formatted_counts = [count[1] for count in counts]  # Extract formatted counts
        results[column].extend(formatted_counts)

# Determine the maximum length of counts
max_length = max(len(counts) for counts in results.values())

# Pad counts lists with empty strings to ensure equal length
for column in results:
    results[column] += [''] * (max_length - len(results[column]))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\n", results_df)


# Apply the function to remove values within ranges less than or equal to the threshold
columns_to_check = ["NS(cm)", "EW(cm)", "UD(cm)"]
print("\nPlease input the threshold values for 'NS(cm)', 'EW(cm)', and 'UD(cm)' columns.")

# Get threshold values from the user
threshold_ns = int(input("Threshold value for 'NS(cm)': "))
threshold_ew = int(input("Threshold value for 'EW(cm)': "))
threshold_ud = int(input("Threshold value for 'UD(cm)': "))

# Apply the function to remove values within ranges less than or equal to the threshold for each column
thresholds = {"NS(cm)": threshold_ns, "EW(cm)": threshold_ew, "UD(cm)": threshold_ud}

# Apply the function to remove values within ranges less than or equal to the threshold for each column
for column, threshold in thresholds.items():
    remove_values_within_threshold(df, column, threshold)

# Save the modified DataFrame to a new CSV file
df.to_csv("OPUS.col", sep='\t', index=False)
df.replace('sagar', '', inplace=True)
df.to_csv("excel.txt", sep='\t', index=False)

df.to_csv("outlier.col", sep='\t', index=False)
