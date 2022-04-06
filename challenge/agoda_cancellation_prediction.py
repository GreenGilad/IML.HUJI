from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import re
import numpy as np
import pandas as pd

features = full_data[["booking_datetime",
                      "checkin_date",
                      "checkout_date",
                      "hotel_city_code",
                      "charge_option",
                      "accommadation_type_name",
                      "hotel_star_rating",
                      "customer_nationality",
                      "guest_is_not_the_customer",
                      "is_first_booking",
                      "cancellation_policy_code",
                      "is_user_logged_in",
                      "original_payment_method",
                      "no_of_adults",
                      "no_of_children",
                      "original_selling_amount",
                      "request_airport"]].to_numpy()

def f16(original_selling_amount: str) -> float:
    return float(float(original_selling_amount) / 5172)

def f17(request_airport : str) -> float:
    return float(request_airport)

def make_condition_to_sum(cond: str, full_price: float, night_price: float) -> float:
    sum = 0
    cond1 = re.split("D", cond)
    days_before_checking = int(cond1[0])
    if cond1[1].find("P"):
        percent = int(re.split("P", cond1[1])[0]) / 100
        sum += full_price * percent * days_before_checking
    else:
        num_nights = int(re.split("N", cond1[1])[0])
        sum += night_price * num_nights * days_before_checking

def f10(cancellation : str, full_price: float, night_price: float) -> tuple(float, float):
    sum = 0
    no_show = 0
    cond = re.split("_", cancellation)
    if len(cond) == 1:
        sum += make_condition_to_sum(cond[0], full_price, night_price)
    else:
        sum += make_condition_to_sum(cond[0], full_price, night_price)
        if cond[1].find("D"):
            sum += make_condition_to_sum(cond[1], full_price, night_price)
        else:
            if cond[1].find("P"):
                percent = int(re.split("P", cond[1])[0]) / 100
                no_show += full_price * percent
            else:
                num_nights = int(re.split("N", cond[1])[0])
                no_show += night_price * num_nights




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
