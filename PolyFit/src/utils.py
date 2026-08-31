"""
Utility functions

"""

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



