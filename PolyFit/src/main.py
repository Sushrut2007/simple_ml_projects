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


#---------------------
# Polynomial Regression
#---------------------

@st.cache_data(show_spinner='Splitting the dataset....')
def perform_train_test_split(df, target):
    """
    Perform train-test split for model training and testing.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    return X_train, X_test, y_train, y_test


@st.cache_data(show_spinner='Converting features...')
def convert_to_poly_features(X_train, X_test, current_degree):
    """
    Transform training and testing features into polynomial features.

    Polynomial features are generated for degrees from 1 through N.
    For each of the current degree, the transformer is fitted only on the training data
    and then used to transform both the training and testing data.

    Returns converted poly features for the current degree model.
    """

    poly_converter = PolynomialFeatures(degree=current_degree, include_bias=False)

    X_train_poly = poly_converter.fit_transform(X_train)
    X_test_poly = poly_converter.transform(X_test)

    return X_train_poly, X_test_poly


@st.cache_data(show_spinner='Training and testing models....')
def poly_fit(data, N):
    """
    Train and evaluate polynomial regression models for multiple degrees.

    The function unpacks the training and testing data, then trains one
    polynomial regression model for each degree from 1 through N. Each model
    is evaluated using the test data and its error metrics are calculated.

    Returns results including degree and associated errors in the form of dictionary.
    """

    results = []
    X_train, X_test, y_train, y_test = data

    # Loop through each degree from 1 to N
    for degree in range(1, N+1):
        # Convert features into poly features
        X_train_poly, X_test_poly = convert_to_poly_features(X_train, X_test, degree)

        # Create model 
        model = LinearRegression()

        # Train and test the dataset
        model.fit(X_train_poly, y_train)

        y_pred = model.predict(X_test_poly)

        # Calculate MAE, RMSE, R2-score
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Store the results for the current degree
        results.append({
            'degree': degree,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        })

    return results


#---------------------
# Polynomial degree evaluation
#---------------------

@st.cache_data(show_spinner="Finding the best degree...")
def find_best_degree(degree_result, tolerance):
    """
    Finds the best degree based on RMSE and provided tolerance.
    
    The function finds the maximum acceptable rmse using tolerane.
    Formula: maximum acceptable rmse = minimum rmse * (1 + tolerance% / 100)

    The next possible best degree will have lesser complexity while having MORE rmse.
    """

    # Sort degree results from lowest to highest rmse
    sorted_degree_result = sorted(degree_result, key=lambda x: x['rmse'])
    # Get the degree with the lowest rmse
    best_degree = sorted_degree_result[0]

    # Apply the formula and find degree with tolerable rmse (if any)
    max_acceptable_rmse = best_degree['rmse'] * (1 + tolerance / 100)

    acceptable_degrees = sorted(
        (d for d in sorted_degree_result
         if d['degree'] < best_degree['degree']
         and d['rmse'] <= max_acceptable_rmse),
        key=lambda d: d['rmse']
    )

    return best_degree, acceptable_degrees
