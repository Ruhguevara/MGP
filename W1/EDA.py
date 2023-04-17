import pandas as pd
import numpy as np
# import string 

class DataExplorer:
    """
    A class for performing exploratory data analysis (EDA) on a Pandas dataframe.
    """

    @staticmethod
    def dqr(df, columns=None):
        # If columns parameter is not specified, use all columns
        if columns is None:
            columns = df.columns.tolist()
        
        # Create DataFrame to store column names
        column_names = pd.DataFrame(columns=['Name'], index=columns, data=columns)
        
        # Create DataFrame to store data types
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
        
        # Create DataFrame to store number of missing values
        num_missing_values = pd.DataFrame(df.isnull().sum(), columns=['# Missing Values'])
        
        # Create DataFrame to store number of present values
        num_present_values = pd.DataFrame(df.count(), columns=['# Present Values'])
        
        # Create DataFrame to store number of unique values
        num_unique_values = pd.DataFrame(df.nunique(), columns=['# Unique Values'])
        
        # Create DataFrame to store minimum and maximum values
        min_max_values = pd.DataFrame(columns=['Min', 'Max'])
        for col in columns:
            try:
                min_max_values.loc[col] = [df[col].min(), df[col].max()]
            except Exception as e:
                print(f"Error calculating min and max for column {col}: {e}")
        
        # Create DataFrame to store frequency of categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            categorical_stats = pd.DataFrame(columns=['Mode', 'Frequency', '# Categories'])
            for col in categorical_cols:
                mode = df[col].mode()[0]
                freq = df[col].value_counts(normalize=True).max()
                n_categories = df[col].nunique()
                categorical_stats.loc[col] = [mode, freq, n_categories]
        else:
            categorical_stats = pd.DataFrame()
            
        # Create DataFrame to store datetime stats
        datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
        if datetime_cols:
            datetime_stats = pd.DataFrame(columns=['Earliest', 'Latest', 'Range'])
            for col in datetime_cols:
                earliest = df[col].min()
                latest = df[col].max()
                date_range = latest - earliest
                datetime_stats.loc[col] = [earliest, latest, date_range]
        else:
            datetime_stats = pd.DataFrame()

        # Concatenate all DataFrames to generate final report
        report = pd.concat([
            column_names,
            data_types,
            num_missing_values,
            num_present_values,
            num_unique_values,
            min_max_values,
            categorical_stats,
            datetime_stats
        ], axis=1)

        return report
    
    @staticmethod
    def generate_data_description_table(df, cmap='GnBu', max_color='#BB0000', mean_color='green'):
        """
        Generates a styled table with descriptive statistics for a Pandas DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to describe.
        cmap (matplotlib colormap): The colormap to use for the background gradient.
        max_color (matplotlib color): The color to use for the maximum value bar.
        mean_color (matplotlib color): The color to use for the mean value bar.

        Returns:
        pd.DataFrame: The styled table with descriptive statistics.
        """
        # Check if input is a Pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError('Input must be a Pandas DataFrame')
        
        # Generate styled table with descriptive statistics
        table = df.describe().T.sort_values(by='max', ascending=False)
        table_style = table.style.background_gradient(cmap=cmap)\
                            .bar(subset=["max"], color=max_color)\
                            .bar(subset=["mean"], color=mean_color)
        
        return table_style





    # @staticmethod
    # def remove_punctuation(text):
    #     """
    #     Removes punctuation from a given string.

    #     Parameters:
    #         text (str): The input string to clean.

    #     Returns:
    #         str: The input string with all punctuation characters removed.
    #     """
    #     return "".join(ch for ch in text if ch not in string.punctuation)

    # @staticmethod
    # def remove_digits(text):
    #     """
    #     Removes digits from a given string.

    #     Parameters:
    #         text (str): The input string to clean.

    #     Returns:
    #         str: The input string with all digits removed.
    #     """
    #     return "".join(ch for ch in text if not ch.isdigit())

    # @staticmethod
    # def remove_whitespace(text):
    #     """
    #     Removes whitespace from a given string.

    #     Parameters:
    #         text (str): The input string to clean.

    #     Returns:
    #         str: The input string with all whitespace characters removed.
    #     """
    #     return "".join(text.split())

    # @staticmethod
    # def replace_text(text, to_replace, replacement):
    #     """
    #     Replaces a given substring in a string with another substring.

    #     Parameters:
    #         text (str): The input string to modify.
    #         to_replace (str): The substring to replace.
    #         replacement (str): The substring to replace it with.

    #     Returns:
    #         str: The input string with all occurrences of `to_replace` replaced by `replacement`.
    #     """
