from load_data import load_dataset
from data_info import dataset_info
from eda import (
    basic_eda, plot_histograms, plot_boxplots,
    plot_correlation_heatmap, plot_pairplot
)
from clean_data import check_missing_values, handle_missing_values, remove_duplicates

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"

columns = [
    "area",
    "perimeter",
    "compactness",
    "length_of_kernel",
    "width_of_kernel",
    "asymmetry_coefficient",
    "length_of_kernel_groove",
    "class"
]

#load dataset
df = load_dataset(url, columns)

# 1) Basic EDA
basic_eda(df, target_col="class")
plot_histograms(df)
plot_boxplots(df)
plot_correlation_heatmap(df)
plot_pairplot(df, hue="class")

# Check missing values
check_missing_values(df)

# Handle missing data using mean or drop
df = handle_missing_values(df, method="mean")

# Remove duplicates if any
df = remove_duplicates(df)

#data information
dataset_info(df)
