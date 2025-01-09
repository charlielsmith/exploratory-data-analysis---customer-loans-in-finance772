import pandas as pd
import sqlalchemy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from data_cleaning import DataTransform
from scipy.stats import zscore

class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def customer_loans_info(self):
        # Call self.df.info() directly to print the info
        self.df.info()

    def customer_loans_descriptive_stats(self):
        # Calculate descriptive statistics
        descriptive_stats = self.df.describe()
        median = self.df.median(numeric_only=True)
        mode = self.df.mode().iloc[0]
        data_range = self.df.max(numeric_only=True) - self.df.min(numeric_only=True)

        # Print descriptives
        print("\nDescriptive Statistics:")
        print(descriptive_stats)
        print("\nMedian:")
        print(median)
        print("\nMode:")
        print(mode)
        print("\nRange:")
        print(data_range)

    def count_distinct_categoricals(self):
        categorical_columns = self.df.select_dtypes(include=['object', 'category'])
        distinct_counts = {col: self.df[col].nunique() for col in categorical_columns.columns}

        print("\nDistinct Value Counts in Categorical Columns`:")
        for col, count in distinct_counts.item():
            print(f"{col}: {count} distinct values")
        
    def shape_of_data(self):
        shape = self.df.shape  # Access the shape attribute (returns a tuple)
        print(f"Dataset has {shape[0]} rows and {shape[1]} columns")


class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_histogram(self, column, bins=50):
        """Plot a histogram for any column."""
        if column in self.df.columns:
            sns.histplot(self.df[column], kde=True, bins=bins)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Column '{column}' not found in the DataFrame.")
    
    def plot_all_boxplots(self):
        """Generate box plots for all numeric columns."""
        numeric_columns = self.df.select_dtypes(include=['number']).columns

        for column in numeric_columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=self.df[column])
            plt.title(f"Box Plot of {column}")
            plt.xlabel(column)
            plt.show()

    def plot_all_numeric_columns(self, bins=50):
        """ Plot histogram for all numeric columns."""
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            print(f"Plotting histogram for {column}")
            self.plot_histogram(column)

    def plot_missing_values(self, before_df=None):
        """
        Plot the number of missing values before and after imputation.
        If before_df is provided, compares before and after imputation.
        """
        # Calculate missing values before and after imputation
        if before_df is not None:
            missing_before = before_df.isnull().sum()
            missing_after = self.df.isnull().sum()
            
            # Plot before vs after imputation for comparison
            missing_comparison = pd.DataFrame({
                'Missing Before': missing_before,
                'Missing After': missing_after
            }).sort_values('Missing Before', ascending=False)

            missing_comparison.plot(kind='bar', figsize=(12, 6))
            plt.title('Comparison of Missing Values Before and After Imputation')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=90)
            plt.show()

        else:
            # Plot the missing values after imputation
            missing_values = self.df.isnull().sum().sort_values(ascending=False)
            missing_values = missing_values[missing_values > 0]
            
            plt.figure(figsize=(12, 6))
            missing_values.plot(kind='bar')
            plt.title('Missing Values After Imputation')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=90)
            plt.show()

# Perform EDA transformation on the data
class DataFrameTransform:
    def __init__(self, df):
        self.df = df
    
    def check_missing_values(self):
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100

        print("\nMissing Values:")
        print(pd.DataFrame({'Missing Count': missing_values, 'Percentage': missing_percentage}))

    def drop_columns_with_missing(self, threshold=30):
        # Drop columns with missing values above the threshold
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        cols_to_drop = missing_percentage[missing_percentage > threshold].index
        self.df.drop(columns=cols_to_drop, inplace=True)
        print(f'\nDropped columns with missing values > {threshold}%: {list(cols_to_drop)}')

    def impute_missing_values(self, strategies):
        """
        Impute missing values for specific columns using the provided strategies.
        :param strategies: A dictionary where keys are column names and values are 'mean, 'median', or 'mode.
        """
        for column, strategy in strategies.items():
            if column in self.df.columns and self.df[column].isnull().sum() > 0:
                if strategy == 'mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                print(f"Imputed missing values in '{column} using {strategy}.")

         # Handle additional columns with missing values manually
         # For 'last_payment_date', use the mode
        if 'last_payment_date' in self.df.columns:
            most_common_date = self.df['last_payment_date'].mode()[0]
            self.df['last_payment_date'].fillna(most_common_date, inplace=True)
            print("Imputed missing values in 'last_payment_date' using mode.")

         # For 'collections_12_mths_ex_med', use the median
        if 'collections_12_mths_ex_med' in self.df.columns:
            median_value = self.df['collections_12_mths_ex_med'].median()
            self.df['collections_12_mths_ex_med'].fillna(median_value, inplace=True)
            print("Imputed missing values in 'collections_12_mths_ex_med' using median.")

        # Handle 'last_credit_pull_date' (using mode)
        if 'last_credit_pull_date' in self.df.columns:
            most_common_date = self.df['last_credit_pull_date'].mode()[0]
            self.df['last_credit_pull_date'].fillna(most_common_date, inplace=True)
            print("Imputed missing values in 'last_credit_pull_date' using mode.")

    def remove_outliers_iqr(self, factor=1.0):
        """
        Remove outliers using the IQR method for all numeric columns.
        
        :param factor: Determines the outlier threshold (default is 1.5 for mild outliers).
        """
        numeric_columns = self.df.select_dtypes(include=['number']).columns

        for column in numeric_columns:
            Q1 = self.df[column].quantile(0.25)  # First quartile (25th percentile)
            Q3 = self.df[column].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range

            lower_bound = Q1 - (factor * IQR)
            upper_bound = Q3 + (factor * IQR)

            # Log the bounds for debugging
            print(f"{column}: Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

            # Filter out rows that fall outside the bounds
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]

        return self.df
    
    def drop_rows_with_missing(self):
        self.df.dropna(inplace=True)
        print("\nDropped rows with any missing values.")


    def log_transform_severe_skew(self):
        """
        Apply log transformation to columns with severe skewness (>2)
        """
        severe_skewed_columns = [
            'annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'out_prncp', 'out_prncp_inv', 
            'total_rec_int', 'total_rec_late_fee', 'recoveries', 
            'collection_recovery_fee', 'last_payment_amount'
        ]

        for column in severe_skewed_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].apply(lambda x: np.log(x) if x > 0 else 0)

                sns.histplot(self.df[column], kde=True)
                plt.title(f"Log-transformed Histogram of {column} (Skew: {self.df[column].skew():.2f})")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.show()

                print(f"New skewness of {column}: {self.df[column].skew():.2f}")
            else:
                print(f"Column '{column}' not found in the DataFrame.")

    def box_cox_transform_moderate_skew(self):
        """
        Apply box cox transformation to columns with moderate skewness (0.5 - 2)
        """
        from scipy.stats import boxcox
        moderate_skewed_columns = [
            'loan_amount', 'funded_amount', 'funded_amount_inv'
        ]
        for column in moderate_skewed_columns:
            if column in self.df.columns:
                # Ensure all values are positive
                if (self.df[column] <= 0).any():
                    adjustment = np.abs(self.df[column].min()) + 1
                    print(f"Adjusting '{column}' by adding {adjustment} to make all values positive.")
                    self.df[column] += adjustment

                # Apply Box-Cox transformation
                try:
                    self.df[column], _ = boxcox(self.df[column])
                    print(f"Applied Box-Cox transformation to '{column}'.")
                except ValueError as e:
                    print(f"Failed to apply Box-Cox transformation to '{column}': {e}")
                    continue

                #Visualise transformed data
                sns.histplot(self.df[column], kde=True)
                plt.title(f"Box-Cox Transformed Histogram of {column} (Skew: {self.df[column].skew():.2f})")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.show()

                print(f"New skewness of {column}: {self.df[column].skew():.2f}")
            else:
                print(f"Column '{column}' not found in the DataFrame.")
    


# Calculate skewness and plot histogram
class DataAnalysis:
    def __init__(self, df):
        self.df = df

    def calculate_and_plot_skew(self, column, bins=50):
        """Calculate skewness of a column and plot its histogram."""
        if column in self.df.columns:
                        # Check if the column is a datetime type and convert to numeric (days from current date)
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                current_date = pd.to_datetime('today')
                self.df[column] = (current_date - self.df[column]).dt.days 

            skewness = self.df[column].skew()
            print(f"Skew of '{column}' column: {skewness}")

            # PLot histogram
            self.df[column].hist(bins=bins)
            plt.title(f"Histogram of {column} (Skew: {skewness:.2f})")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Column '{column}' not found in the DataFrame.")

if __name__ == "__main__":
    # Load CSV into a DataFrame
    data_frame = pd.read_csv('loan_payments.csv')

    # Initialize classes
    df_info = DataFrameInfo(data_frame)
    df_transform = DataFrameTransform(data_frame)
    plotter = Plotter(data_frame)

    # Step 1: Print basic dataset information
    print("Step 1: Dataset Information")
    df_info.customer_loans_info()
    df_info.customer_loans_descriptive_stats()

    # Step 2: Visualize numeric columns for outliers
    print("\nStep 2: Visualizing all numeric columns for outliers...")
    plotter.plot_all_numeric_columns()

    # Step 3: Handle missing values
    print("\nStep 3: Handling missing values")
    imputation_strategies = {
        'funded_amount': 'mean',
        'term': 'mode',
        'int_rate': 'median',
        'employment_length': 'mode',
    }
    df_transform.impute_missing_values(imputation_strategies)

    # Step 4: Plot missing values before and after imputation
    print("\nStep 4: Plotting missing values before and after imputation")
    plotter.plot_missing_values(before_df=data_frame)

    # Step 5: Drop columns with excessive missing values
    print("\nStep 5: Dropping columns with >30% missing values")
    df_transform.drop_columns_with_missing(threshold=30)

    # Step 6: Final check for missing values
    print("\nStep 6: Final missing values check")
    df_transform.check_missing_values()

    # Step 7: Perform transformations
    print("\nStep 7: Performing transformations")
    df_transform.convert_verification_status()
    df_transform.transform_skewed_columns()

    # Step 8: Save the cleaned and transformed DataFrame
    print("\nStep 8: Saving cleaned and transformed dataset")
    data_frame.to_csv('cleaned_data.csv', index=False)

    print("Data cleaning, transformation, and visualization completed!")



