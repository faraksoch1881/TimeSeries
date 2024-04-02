import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy import stats  # Import stats module from scipy
import matplotlib.patches as mpatches

# Replace 'file_path' with the path to your OPUS.col file
# Get the current working directory
current_directory = os.getcwd()

# Find files ending with ".col" in the current directory
col_files = glob.glob(os.path.join(current_directory, "*.col"))
if os.path.basename(col_files[0]) == "OPUS.col":
    data_first = pd.read_csv(col_files[0], delimiter='\t')
    data_second = pd.read_csv(col_files[1], delimiter='\s+')
else:
    data_first = pd.read_csv(col_files[1], delimiter='\t')
    data_second = pd.read_csv(col_files[0], delimiter='\s+')


north_second = data_second.iloc[:, 1]
east_second = data_second.iloc[:, 2]
up_second = data_second.iloc[:, 3]

# Function to plot and perform linear regression
def plot_and_regression(ax, x, y, title, color, regression_color):
    ax.scatter(x, y, color=color,  s=10)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', color=regression_color)

    ax.text(-0.1, 0.5, title, transform=ax.transAxes, fontsize=10, verticalalignment='center', rotation='vertical')
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['bottom'].set_linewidth(0.5)  # Set the linewidth of the bottom spine


# Read the first .col file



# Drop rows where the second, third, or fourth column is null
data_first_second_column = data_first.dropna(subset=[data_first.columns[1]])
data_first_third_column = data_first.dropna(subset=[data_first.columns[2]])
data_first_fourth_column = data_first.dropna(subset=[data_first.columns[3]])

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
# Add padding around the figure
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)



for i, (ax, data_column, color) in enumerate(zip(axs, [data_first_second_column, data_first_third_column, data_first_fourth_column], ['blue']*3)):
    x = data_column.iloc[:, 0]
    y = data_column.iloc[:, i + 1]
    title = data_column.columns[i + 1]  # Extracting the title based on the index

    # Extracting the year from x values
    year = x.astype(str).str[:4]

    # Grouping data by year and performing linear regression
    slopes = []
    square_r_values = []  # Define square_r_values list here
    for yr, group in zip(year.unique(), [group for _, group in data_column.groupby(year)]):
        group_x = group.iloc[:, 0]
        group_y = group.iloc[:, i + 1]
        z = np.polyfit(group_x, group_y, 1)
        p = np.poly1d(z)
        slope, intercept, r_value, _, _ = stats.linregress(group_x, group_y)
        slopes.append(z[0])
        square_r_values.append(r_value ** 2)

    # Calculating the average slope
    average_slope = np.mean(slopes)
    average_square_r = np.mean(square_r_values)

    plot_and_regression(ax, x, y, title, color, 'black')
    ax.text(0.05, 0.95, f'OPUS = {average_slope:.2f} cm/yr\nR(squared) = {average_square_r*100:.2f}%', transform=ax.transAxes, fontsize=10, verticalalignment='top')



    # Set x-axis limits to start from the minimum value of x
    if i == 0:

        combined_y = pd.concat([north_second, y])
    elif i == 1:
        combined_y = pd.concat([east_second, y])
    else:
        combined_y = pd.concat([up_second, y])

    min_x = min(x)
    max_x = max(x)
    x_range = max_x - min_x
    ax.set_xlim(min_x - 0.1 * x_range, max_x + 0.1 * x_range)

    min_y = min(combined_y)
    max_y = max(combined_y)
    y_range = max_y - min_y

    # Set the limits for the y-axis with a 10% padding
    ax.set_ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)

    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.spines['left'].set_linewidth(0.5)  # Set the linewidth of the left spine

    # Calculate and display slope and R-squared for data_second
    slopes_data_second = []
    r_values_data_second = []
    for yr, group in zip(year.unique(), [group for _, group in data_second.groupby(year)]):
        group_x = group.iloc[:, 0]
        group_y = group.iloc[:, i + 1]  # Corrected to use 'i' as the index for the y-values
        slope, _, r_value, _, _ = stats.linregress(group_x, group_y)
        slopes_data_second.append(slope)
        r_values_data_second.append(r_value)

    # Calculate the average slope and average R-value for data_second
    avg_slope_data_second = np.mean(slopes_data_second)
    avg_r_value_data_second = np.mean(r_values_data_second)

    # Add text to subplot displaying the average slope and average R-value for data_second
    ax.text(0.25, 0.05, f'PPP: {avg_slope_data_second:.2f} cm/yr\nR-value: {avg_r_value_data_second*100:.2f}%', transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right')



# Initialize lists to store slopes and square R values for each column
slopes_data_second = []
square_r_values_data_second = []

# Extract the x values (first column) for data_second
x_data_second = data_second.iloc[:, 0]

# Loop through the second, third, and fourth columns
for i in range(1, 4):  # Start from index 1 since index 0 is the x-axis column
    y_data_second = data_second.iloc[:, i]  # Extract y values for the current column
    
    # Perform linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x_data_second, y_data_second)
    
    # Append slope and square R value to respective lists
    slopes_data_second.append(slope)
    square_r_values_data_second.append(r_value ** 2)


# Overlay data from the second dataframe onto the existing subplots
for i, (ax, color) in enumerate(zip(axs, ['red']*3)):
    x = data_second.iloc[:, 0]
    y = data_second.iloc[:, i + 1]
    plot_and_regression(ax, x, y, '', color, 'black')

# Define custom legend patches
red_patch = mpatches.Patch(color='red', label='PPP')
blue_patch = mpatches.Patch(color='blue', label='OPUS')

# Add legend with custom patches
fig.legend(handles=[red_patch, blue_patch])
# Show and save plot as PDF
plt.tight_layout(pad=3.0)
plt.savefig(f'linear_regression_plots.pdf')
plt.show()
