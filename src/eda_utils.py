import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style for better aesthetics
sns.set_theme(style="dark")

def plot_correlation_heatmap(df, numerical_cols, figsize=(12, 8)):
    """Plot a correlation heatmap for numerical columns."""
    if numerical_cols:
        plt.figure(figsize=figsize)
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

def plot_histograms(df, numerical_cols, figsize=(12, 8)):
    """Plot histograms for numerical columns."""
    if numerical_cols:
        df[numerical_cols].hist(bins=20, figsize=figsize, layout=(len(numerical_cols), 1))
        plt.suptitle('Histograms of Numerical Features', y=1.02)
        plt.tight_layout()
        plt.show()

def plot_box_plots(df, numerical_cols, figsize=(12, 8)):
    """Plot individual box plots for numerical columns."""
    if numerical_cols:
        for col in numerical_cols:
            plt.figure(figsize=figsize)
            sns.boxplot(data=df, y=col)
            plt.title(f'Box Plot of {col}')
            plt.show()

def plot_count_plots(df, categorical_cols, figsize=(12, 8)):
    """Plot count plots for categorical columns."""
    if categorical_cols:
        for col in categorical_cols:
            if df[col].nunique() > 20:  # Skip columns with too many unique values
                print(f"Skipping count plot for {col} (too many unique values)")
                continue
            plt.figure(figsize=figsize)
            sns.countplot(data=df, x=col)
            plt.title(f'Count Plot of {col}')
            plt.xticks(rotation=45)
            plt.show()

def plot_feature_target_relationships(df, target_col, numerical_cols, categorical_cols, figsize=(12, 8)):
    """Plot feature-target relationships."""
    if target_col:
        if target_col in categorical_cols:
            for col in numerical_cols:
                plt.figure(figsize=figsize)
                sns.boxplot(data=df, x=target_col, y=col)
                plt.title(f'Box Plot of {col} by {target_col}')
                plt.xticks(rotation=45)
                plt.show()
        elif target_col in numerical_cols:
            for col in numerical_cols:
                if col != target_col:
                    plt.figure(figsize=figsize)
                    sns.scatterplot(data=df, x=col, y=target_col)
                    plt.title(f'Scatter Plot of {col} vs {target_col}')
                    plt.show()

def plot_pair_plot(df, numerical_cols):
    """Plot pair plots for numerical columns."""
    if numerical_cols and len(numerical_cols) <= 5:  # Limit to avoid slow rendering
        sns.pairplot(df[numerical_cols])
        plt.suptitle('Pair Plot of Numerical Features', y=1.02)
        plt.show()

        
# Example usage
if __name__ == "__main__":
    # Sample dataset (replace with your own)
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical([iris.target_names[i] for i in iris.target])

    # Define numerical and categorical columns (optional, auto-detected if omitted)
    numerical_cols = iris.feature_names
    categorical_cols = ['species']

    # Run EDA
    plot_eda(df, target_col='species', numerical_cols=numerical_cols, categorical_cols=categorical_cols)