from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator

import numpy as np
import pandas as pd


def check_validation(full_data):
    checkin_date_valid = full_data[full_data['booking_datetime'] <= full_data['checkin_date']]
    checkout_date_valid = checkin_date_valid[checkin_date_valid['checkin_date'] < checkin_date_valid['checkout_date']]
    cancellation_validation = checkout_date_valid[
        checkout_date_valid['cancellation_datetime'] <= checkout_date_valid['checkout_date']]

    return cancellation_validation


def refactor_by_challenge_request(data_after_validation):
    checkout_date = data_after_validation['2018-07-02' < data_after_validation['checkout_date'] < '2018-09-30']
    booking_date = checkout_date['2017-07-02' < checkout_date['booking_date'] < '2018-09-30']
    return booking_date


def load_data(filename: str):
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
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    # print(full_data.size)
    # clean_data = check_validation(full_data)
    # print(clean_data.shape)
    data_after_filtration = refactor_by_challenge_request
    features = full_data[["h_booking_id",
                          "hotel_id",
                          "accommadation_type_name",
                          "hotel_star_rating",
                          "customer_nationality"]]
    labels = full_data["cancellation_datetime"]

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


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
