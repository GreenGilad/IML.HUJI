import datetime
import re
import csv
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from challenge.Currency import RealTimeCurrencyConverter
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
# from __future__ import annotations
# from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

url = 'https://api.exchangerate-api.com/v4/latest/USD'
converter = RealTimeCurrencyConverter(url)


def make_condition_to_sum(cond: str, full_price: float,
                          night_price: float) -> float:
    sum = 0
    cond1 = re.split("D", cond)
    days_before_checking = int(cond1[0])
    if cond1[1].find("P") != -1:
        percent = int(re.split("P", cond1[1])[0]) / 100
        sum += full_price * percent * days_before_checking
    else:
        num_nights = int(re.split("N", cond1[1])[0])
        sum += night_price * num_nights * days_before_checking
    return sum


def f10(cancellation: str, full_price: float, night_price: float) -> (float, float):
    if cancellation == "UNKNOWN":
        return 0, 0
    sum = 0
    no_show = 0
    cond = re.split("_", cancellation)
    if len(cond) == 1:
        sum += make_condition_to_sum(cond[0], full_price, night_price)
    else:
        sum += make_condition_to_sum(cond[0], full_price, night_price)
        if cond[1].find("D") != -1:
            sum += make_condition_to_sum(cond[1], full_price, night_price)
        else:
            if cond[1].find("P") != -1:
                percent = int(re.split("P", cond[1])[0]) / 100
                no_show += full_price * percent
            else:
                num_nights = int(re.split("N", cond[1])[0])
                no_show += night_price * num_nights
    return sum, no_show


def get_cancellation(features: pd.DataFrame):
    sum = []
    no_show = []
    for index, row in features.iterrows():
        a, b = f10(row.cancellation_policy_code, row.original_selling_amount, row.price_per_night)
        sum.append(a)
        no_show.append(b)
    return sum, no_show


def change_currency(features: pd.DataFrame):
    total = []
    for index, row in features.iterrows():
        total_amount = converter.convert(row.original_payment_currency, 'USD', row.original_selling_amount)
        total.append(total_amount)
    return total


def load_data(filename: str, with_lables = True):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()

    features = full_data[["booking_datetime",
                          "checkin_date",
                          "checkout_date",
                          "hotel_city_code",
                          "hotel_star_rating",
                          "charge_option",
                          "accommadation_type_name",
                          "guest_is_not_the_customer",
                          "cancellation_policy_code",
                          "is_user_logged_in",
                          "original_payment_method",
                          "no_of_adults",
                          "no_of_children",
                          "original_selling_amount",
                          "customer_nationality",
                          "original_payment_type",
                          "original_payment_currency"]]

    features["checkin_date"] = pd.to_datetime(features["checkin_date"])
    features["checkout_date"] = pd.to_datetime(features["checkout_date"])
    features["booking_datetime"] = pd.to_datetime(features["booking_datetime"])
    features["duration"] = (features["checkout_date"] - features["checkin_date"]).dt.days.astype(int)
    features['checkin_date_day_of_year'] = (features['checkin_date'].dt.dayofyear).astype(int)
    features["booking_hour"] = (pd.DatetimeIndex(features['booking_datetime']).hour).astype(int)
    selling_amount = change_currency(features)
    features["original_selling_amount"] = selling_amount
    features["price_per_night"] = (features["original_selling_amount"] / features["duration"])

    # fixing dummies features
    features = pd.get_dummies(features, prefix="hotel_star_rating_", columns=["hotel_star_rating"])
    features = pd.get_dummies(features, prefix="accommadation_type_name_", columns=["accommadation_type_name"])
    features = pd.get_dummies(features, prefix="charge_option_", columns=["charge_option"])
    features = pd.get_dummies(features, prefix="customer_nationality_", columns=["customer_nationality"])
    features = pd.get_dummies(features, prefix="no_of_adults_", columns=["no_of_adults"])
    features = pd.get_dummies(features, prefix="no_of_children_", columns=["no_of_children"])
    features = pd.get_dummies(features, prefix="original_payment_type_", columns=["original_payment_type"])
    features = pd.get_dummies(features, prefix="original_payment_method_", columns=["original_payment_method"])
    features = pd.get_dummies(features, prefix="hotel_city_code_", columns=["hotel_city_code"])
    features[features["is_user_logged_in"] == "FALSE"] = 0
    features[features["is_user_logged_in"] == "TRUE"] = 1

    features["cancellation_sum"], features["cancellation_no_show"] = get_cancellation(features)

    # removing old features
    for f in ["checkout_date", "booking_datetime", "checkin_date", "cancellation_policy_code",
              "original_payment_currency"]:
        features.drop(f, axis=1, inplace=True)

    labels = None

    if with_lables:
        # making label_for_regression
        labels = full_data["cancellation_datetime"]
        labels = pd.to_datetime(labels.fillna(pd.Timestamp('21000101')))
        # label_for_regression = (labels.dt.date - features["booking_datetime"].dt.date).dt.days.astype(int)

        # making 0/1 labels
        labels = full_data["cancellation_datetime"]
        labels = labels.fillna(0)
        labels[labels != 0] = 1

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


def make_same_features(train_X: pd.DataFrame, test_X: pd.DataFrame):
    new_test = pd.DataFrame(np.zeros((test_X.shape[0], 0)))
    new_train = pd.DataFrame(np.zeros((train_X.shape[0], 0)))

    train_features = train_X.columns.values
    test_features = test_X.columns.values
    new_features = []

    for f in train_features:
        if f in test_features:
            new_features.append(f)

    new_features = np.array(new_features)
    for f in new_features:
        new_test[f] = test_X[f]
        new_train[f] = train_X[f]
    return new_train, new_test

if __name__ == '__main__':
    np.random.seed(0)
    for_test = False
    if for_test:
        # Load data
        df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
        # train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels, random)
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels, random_state=0)
        # train_X, train_y, test_X, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels, random_state=0)

        # Fit model over data
        estimator = AgodaCancellationEstimator().fit(train_X.to_numpy(), train_y.to_numpy().astype(int))

        # Store model predictions over test set
        evaluate_and_export(estimator, test_X, r"C:\Users\User\Documents\CSE_2\IML\dataChallenge\208881136_207029398_207543620_validation.csv")

        y_pred_lr = estimator.predict(test_X)

        acc_lr = accuracy_score(test_y.to_numpy().astype(int), y_pred_lr)
        conf = confusion_matrix(test_y.to_numpy().astype(int), y_pred_lr)
        clf_report = classification_report(test_y.to_numpy().astype(int), y_pred_lr)

        print(f"Accuracy Score of Logistic Regression is : {acc_lr}")
        print(f"Confusion Matrix : n{conf}")
        print(f"Classification Report : n{clf_report}")

    if not for_test:
        train_X, train_y = load_data("../datasets/agoda_cancellation_train.csv")
        test_x, test_y = load_data(
            r"week_5_test_data.csv", False)

        # the test dataset may have different features then the test dataset
        train_X, test_x = make_same_features(train_X, test_x)

        # Fit model over data
        estimator = AgodaCancellationEstimator().fit(train_X.to_numpy(),
                                                     train_y.to_numpy().astype(
                                                         int))

        # Store model predictions over test set
        evaluate_and_export(estimator, test_x.to_numpy(),
                            r"C:\Users\feino\Documents\iml\IML.HUJI\challenge\208881136_207029398_207543620_week5.csv")



