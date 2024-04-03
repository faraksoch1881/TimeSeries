import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy import stats  # Import stats module from scipy

# Replace 'file_path' with the path to your OPUS.col file
# Get the current working directory
current_directory = os.getcwd()

# Find files ending with ".col" in the current directory
col_files = glob.glob(os.path.join(current_directory, "*.col"))

# Use the first found file as file_path
if col_files:
    file_path = col_files[0]
else:
    raise FileNotFoundError("No .col files found in the current directory")

# Read the file using pandas
data = pd.read_csv(file_path, delimiter='\t')

# Drop rows where the fourth column is null
data_fourth_column = data.dropna(subset=[data.columns[3]])

# Function to plot and perform linear regression
def plot_and_regression(ax, x, y, title):
    # Extracting the year from x values
    years = x.astype(str).str[:4].unique()

    # Define colors for scatter plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))  # Choose a colormap and generate colors

    # Grouping data by year and plotting scatter points with different colors
    for i, year in enumerate(years):
        group_indices = (x.astype(str).str[:4] == year)
        group_x = x[group_indices]
        group_y = y[group_indices]
        ax.scatter(group_x, group_y, color=colors[i], label=year, s=6)

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'r--', label='Linear Regression')
        ax.text(-0.1, 0.5, title, transform=ax.transAxes, fontsize=10, verticalalignment='center', rotation='vertical')
        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_linewidth(0.5)  # Set the linewidth of the bottom spine


# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

data_column = data_fourth_column
x = data_column.iloc[:, 0]
y = data_column.iloc[:, 3]  # Use the fourth column
title = data_column.columns[3]  # Extracting the title based on the index
# Extracting the year from x values
year = x.astype(str).str[:4]

# Grouping data by year and performing linear regression
slopes = []
square_r_values = []  # Define square_r_values list here
for yr, group in zip(year.unique(), [group for _, group in data_column.groupby(year)]):
    group_x = group.iloc[:, 0]
    group_y = group.iloc[:, 3]  # Use the fourth column
    z = np.polyfit(group_x, group_y, 1)
    p = np.poly1d(z)
    slope, intercept, r_value, std_err, _ = stats.linregress(group_x, group_y)
    slopes.append(z[0])
    square_r_values.append(r_value ** 2)

# Calculating the average slope
std_err *= 1.96
average_slope = np.mean(slopes)
average_square_r = np.mean(square_r_values)
average_std_err = np.mean(std_err)

plot_and_regression(ax, x, y, title)
ax.text(0.05, 0.95, f'Displacement = {average_slope:.2f} ± {average_std_err:.2f} cm/yr\nR(squared) = {average_square_r*100:.2f}%', transform=ax.transAxes, fontsize=22, verticalalignment='bottom')

# Set x-axis limits to start from the minimum value of x
min_x = min(x)
max_x = max(x)
x_range = max_x - min_x
ax.set_xlim(min_x - 0.1 * x_range, max_x + 0.1 * x_range)

min_y = min(y)
max_y = max(y)
y_range = max_y - min_y

# Set the limits for the y-axis with a 10% padding
ax.set_ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)

ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)
ax.spines['left'].set_linewidth(0.5)  # Set the linewidth of the left spine

# Show and save plot as PNG
plt.tight_layout(pad=3.0)
plt.savefig(file_path + "_scatter.png")
plt.show()
