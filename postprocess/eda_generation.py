import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

class EDA(object):
    def __init__(self, data: pd.DataFrame, save_dir: str):
        """
        :param data: original dataset
        :param save_dir: path to save the EDA plot
        """
        self.data = data
        # Identify categorical features
        self.categorical_features = data.select_dtypes(include=['object', 'category']).columns

    def plot_dist(self):
        df = self.data.copy()
        # Number of features and set the number of plots per row
        num_features = len(df.columns)
        plots_per_row = 5
        num_rows = (num_features + plots_per_row - 1) // plots_per_row  # Calculate number of rows needed

        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(plots_per_row * 5, num_rows * 4))

        # Flatten the axes array for easy indexing
        axes = axes.flatten()

        # Plot histogram for each feature
        for i, feature in enumerate(df.columns):
            sns.histplot(df[feature], ax=axes[i], bins=10, kde=True)
            axes[i].set_title(feature)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        save_path_dist = os.path.join(self.save_dir, 'eda_dist.jpg')
        plt.savefig(fname=save_path_dist, dpi=1000)

        return save_path_dist

    def desc_dist(self):
        df = self.data.copy()
        # Generate descriptive statistics for numerical features
        numerical_stats = df.describe()

        # Analyze categorical features
        categorical_analysis = {feature: df[feature].value_counts() for feature in self.categorical_features}
        analysis_input = "Numerical Features:\n"

        for feature in df.select_dtypes(include='number').columns:
            mean = numerical_stats.loc['mean', feature]
            median = numerical_stats.loc['50%', feature]
            std_dev = numerical_stats.loc['std', feature]
            min_val = numerical_stats.loc['min', feature]
            max_val = numerical_stats.loc['max', feature]

            analysis_input += (
                f"Feature: {feature}\n"
                f"Mean: {mean:.2f}, Median: {median:.2f}, Standard Deviation: {std_dev:.2f}\n"
                f"Min: {min_val}, Max: {max_val}\n\n"
            )

        analysis_input += "\nCategorical Features:\n"

        for feature in self.categorical_features:
            counts = df[feature].value_counts()
            analysis_input += f"Feature: {feature}\n{counts}\n\n"

        return analysis_input

    def plot_corr(self):
        df = self.data.copy()

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()
        # Convert categorical features using label encoding
        for feature in self.categorical_features:
            df[feature] = label_encoder.fit_transform(df[feature])

        # Generate scatter plots for each pair of features
        plt.figure(figsize=(8, 8))
        sns.pairplot(df)
        # Save the plot
        save_path_scat = os.path.join(self.save_dir, 'eda_scat.jpg')
        plt.savefig(fname=save_path_scat, dpi=1000)

        # Calculate the correlation matrix
        correlation_matrix = df.corr()
        # Create a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='PuBu', fmt=".2f", square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap')
        # Save the plot
        save_path_corr = os.path.join(self.save_dir, 'eda_corr.jpg')
        plt.savefig(fname=save_path_corr, dpi=1000)

        return correlation_matrix, save_path_scat, save_path_corr

    def desc_corr(self, correlation_matrix, threshold=0.5):
        """
        :param correlation_matrix: correlation matrix of the original dataset
        :param threshold: correlation below the threshold will not be included in the summary
        :return: string of correlation_summary
        """
        # Prepare a summary of significant correlations
        correlation_summary = ""

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    correlation_summary += (
                        f"Correlation between {correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: "
                        f"{correlation_matrix.iloc[i, j]:.2f}\n"
                    )
        return correlation_summary

    def generate_eda(self):
        plot_path_dist = self.plot_dist()
        corr_mat, plot_path_scat, plot_path_corr = self.plot_corr()

        dist_analysis = self.desc_dist()
        corr_analysis = self.desc_corr(corr_mat)

        eda_result = {'plot_path_dist': plot_path_dist,
                      'plot_path_scat': plot_path_scat,
                      'plot_path_corr': plot_path_corr,
                      'dist_analysis': dist_analysis,
                      'corr_analysis': corr_analysis}

        return eda_result






