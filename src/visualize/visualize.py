import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.getcwd())
from src.data.make_dataset import load_df
from definitions import ROOT_DIR

plot_folder = os.path.join(ROOT_DIR, 'reports/figures/')
boxplot_path = os.path.join(plot_folder, 'boxplots.png')
distributions_path = os.path.join(plot_folder, 'distributions.png')
correlations_path = os.path.join(plot_folder, 'correlations.png')


def plot_distributions(df, path):
    """
    Generate and save histograms for various columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
        None
    """
    print("Generating distributions")
    nrows, ncols = 3, 3
    _, axis = plt.subplots(nrows, ncols, figsize=(20, 10))
    columns = ['length_diff', 'similarity', 'reference_length',
               'translation_length', 'length_diff', 'ref_tox',
               'trn_tox'
               ]
    labels = [
        'Length difference', 'Similarity',
        'Reference length', 'Translation length',
        'Difference in translation and reference lengths',
        'Reference toxicity', 'Translation toxicity'
    ]
    for i in range(len(columns)):
        ax = axis[i//ncols][i % ncols]
        sns.histplot(df[columns[i]], bins=40, kde=True, ax=ax)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Frequency')
        ax.set_title(f"{labels[i]} distribution")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_boxplots(df, path):
    """
    Generate and save boxplots for selected columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
        None
    """
    print("Generating boxplots")
    _, axs = plt.subplots(2, 2, figsize=(6, 6))
    data = [df['similarity'], df['length_diff'], df['ref_tox'], df['trn_tox']]
    labels = ['Similarity', 'Length Difference',
              'Reference Toxicity', 'Translation Toxicity']
    for i in range(2):
        for j in range(2):
            axs[i, j].boxplot(data[i * 2 + j])
            axs[i, j].set_ylabel('Values')
            axs[i, j].set_title(labels[i * 2 + j])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_correlations(df, path):
    """
    Generate and save a correlation matrix heatmap for selected columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
        None
    """
    print("Generating correlation matrix")
    correlation_matrix = df[['similarity',
                             'length_diff', 'ref_tox', 'trn_tox']].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(path)
    plt.close()


def visualize(df):
    """
    Generate and save various exploratory visualizations based on the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
        tuple: A tuple of file paths where the generated visualizations are saved, including
        distributions_path, boxplot_path, and correlations_path.
    """
    os.makedirs(plot_folder, exist_ok=True)
    plot_distributions(df, distributions_path)
    plot_boxplots(df, boxplot_path)
    plot_correlations(df, correlations_path)
    
    return distributions_path, boxplot_path, correlations_path


def main():
    df = load_df()
    visualize(df)


if __name__ == "__main__":
    main()
