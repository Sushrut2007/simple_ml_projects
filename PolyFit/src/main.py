"""
Dataset preprocessing and ml model related functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


"""
Polynomial Regression
"""

@st.cache_data
def perform_train_test_split(df, target):
    """
    Perform train-test split for model training and testing.
    """
    X = df.drop(columns=[target])
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    return X_train, X_test, y_train, y_test
