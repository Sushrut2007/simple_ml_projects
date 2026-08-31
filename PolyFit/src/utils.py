"""
Utility functions

"""

import numpy as np
import pandas as pd


# -----------------------------
# Data loading
#------------------------------

def load_dataset(filename):
    """
    Loads a CSV file into a dataframe.
    """
    df = pd.read_csv(filename)

    return df


# -----------------------------
# Dataset validation
#------------------------------

def validate_dataset(df):
    """
    Validate the dataset against the requirements of Polynomial Regression.

    The function passes through multiple checks:
        1. Check if the df contains other than numeric values.
        2. Check if the df contains atleast 2 columns.
    
    Return True for valid dataset.
    """

    # Check if the df contains other than numeric values
    allowed_dtypes = ['int64', 'int32', 'float64', 'float32', 'category']
    df_dtypes = df.dtypes

    # Check if the df contains atleast 2 columns
    num_cols = df.shape[1]

    if not all(str(dtype) in allowed_dtypes for dtype in df_dtypes):
        return "Please use a valid dataset!"
    
    return True