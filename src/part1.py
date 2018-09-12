# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import pandas as pd
import numpy as np
import datetime
import random

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.1  # step size
max_iters = 100  # max iterations

models = {
    "linreg": LinearRegression(),
    "knn": KNeighborsRegressor(),
    "ridge": Ridge(),
    "dtr": DecisionTreeRegressor(),
    "rfr": RandomForestRegressor(),
    "gbr": GradientBoostingRegressor(),
    "sgdr": SGDRegressor(),
    "svr": SVR(),
    "linsvr": LinearSVR(),
    "mlpr": MLPRegressor(),
}


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
    df = pd.read_csv("../data/part1/diamonds.csv")
    return df


def data_preprocess(data):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param data: the given dataset (format: panda DataFrame)
    :return: train_data       train data contains only inputs
             train_labels     train data contains only labels
             test_data        test data contains only inputs
             test_labels      test data contains only labels
             train_data_full       train data (full) contains both inputs and labels
             test_data_full       test data (full) contains both inputs and labels
    """
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=train_test_split_test_size, random_state=seed)

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["price"], axis=1)
    train_labels = train_data_full["price"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["price"], axis=1)
    test_labels = test_data_full["price"]
    train_data = pd.get_dummies(train_data, columns=['cut', 'color', 'clarity'])
    test_data = pd.get_dummies(test_data, columns=['cut', 'color', 'clarity'])
    # Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # Tricks: add dummy intercept to both train and test
    train_data['intercept_dummy'] = pd.Series(1.0, index=train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index=test_data.index)
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


if __name__ == '__main__':
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    data = load_data()

    # Step 2: Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)

    # Step 3: Learning Start
    for model in models:
        clf = models[model]
        start_time = datetime.datetime.now()  # Track learning starting time
        clf.fit(train_data.values, train_labels.values)
        end_time = datetime.datetime.now()  # Track learning ending time

        exection_time = (end_time - start_time).total_seconds()  # Track execution time
        predictions = clf.predict(test_data.values)

        # Step 4: Results presentation
        print(clf)
        print("Learn: execution time={t:.3f} seconds".format(t=exection_time))

        # Build baseline model
        print("R2:", r2_score(test_labels.values, predictions))  # R2 should be maximize
        print("MSE:", mean_squared_error(test_labels.values, predictions))
        print("RMSE:", np.sqrt(mean_squared_error(test_labels.values, predictions)))
        print("MAE:", mean_absolute_error(test_labels.values, predictions), "\n")
