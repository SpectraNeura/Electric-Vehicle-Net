import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_correlation_matrix(df, title):
    """
    Renames columns to English, computes the correlation matrix for numeric columns,
    prints the correlation values, and plots a heatmap with annotated numbers.
    """
    # Rename columns from Korean to English
    rename_dict = {
        '주행거리(km)': 'Mileage (km)',
        '보증기간(년)': 'Warranty (years)',
        '연식(년)': 'Age (years)',
        '배터리용량': 'Battery Capacity',
        '가격(백만원)': 'Price (M KRW)'
    }
    df = df.rename(columns=rename_dict)

    # Define the numeric columns for analysis
    numeric_cols = ['Mileage (km)', 'Warranty (years)', 'Age (years)', 'Battery Capacity']
    if 'Price (M KRW)' in df.columns:
        numeric_cols.append('Price (M KRW)')

    # Calculate correlation matrix
    corr = df[numeric_cols].corr()

    # Print the correlation matrix
    print(title)
    print(corr)
    print("\n")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, interpolation='none', cmap='coolwarm')
    plt.colorbar(im)

    # Set x and y ticks
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)

    # Annotate each cell with the correlation value
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            text = f"{corr.iloc[i, j]:.2f}"
            plt.text(j, i, text, ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()


# Directory where the CSV files are stored
data_dir = "data"

# Mapping of CSV filenames to their corresponding imputation methods
file_methods = {
    "train_imputed_model.csv": "Model-based Imputation",
    "train_imputed_mean.csv": "Mean Imputation",
    "train_imputed_knn.csv": "KNN Imputation",
    "train_imputed_linear.csv": "Linear Interpolation",
    "train_imputed_poly.csv": "Polynomial Interpolation"
}

# Process each file: read CSV, compute & plot correlation matrix with title, and annotate values.
for filename, method in file_methods.items():
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    plot_correlation_matrix(df, f"Correlation Matrix: {method}")
