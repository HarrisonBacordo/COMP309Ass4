# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import pandas as pd
import numpy as np
import datetime
import random

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations

models = {
    "knn": KNeighborsClassifier(),
    "nbc": GaussianNB(),
    "svc": SVC(),
    "dtc": DecisionTreeClassifier(),
    "rfc": RandomForestClassifier(),
    "abc": AdaBoostClassifier(),
    "gbc": GradientBoostingClassifier(),
    "ldac": LinearDiscriminantAnalysis(),
    "mlpc": MLPClassifier(),
    "lrc": LogisticRegression(),
}


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
    df_train = pd.read_csv("../data/part3/adult.data")
    df_test = pd.read_csv("../data/part3/adult.test", skiprows=[0])
    return df_train, df_test


def data_preprocess(train, test):
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
    train_data, test_data = train, test
    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.iloc[:, :-1]
    train_labels = train_data_full.iloc[:, -1]

    test_data_full = test_data.copy()
    test_data = test_data.iloc[:, :-1]
    test_labels = test_data_full.iloc[:, -1]
    train_data = pd.get_dummies(train_data, columns=train_data.select_dtypes(include='object').columns)
    test_data = pd.get_dummies(test_data, columns=test_data.select_dtypes(include='object').columns)
    imputer = Imputer(strategy="median")
    train_data = pd.DataFrame(imputer.fit_transform(train_data))
    test_data = pd.DataFrame(imputer.transform(test_data))
    # Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


if __name__ == '__main__':
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    train, test = load_data()

    # Step 2: Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(train, test)

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
        print("ACCURACY:", accuracy_score(test_labels.values, predictions))  # R2 should be maximize
        print("F1:", f1_score(test_labels.values, predictions))
        print("PRECISION:", precision_score(test_labels.values, predictions))
        print("RECALL:", recall_score(test_labels.values, predictions), "\n")
        print("ROC:", roc_curve(test_labels.values, predictions), "\n")
        print("ROC_AUC:", roc_auc_score(test_labels.values, predictions), "\n")
