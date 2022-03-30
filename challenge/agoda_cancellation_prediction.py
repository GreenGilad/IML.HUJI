import re
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from challenge.agoda_cancellation_estimator import *
from sklearn import metrics
import plotly as plt



def f0(datetime: str):
    day = int(datetime[8:10])
    month = int(datetime[5:7])
    hour = int(datetime[11:13])
    return float(hour / 24)





def f_3(city_code: str) -> float:
    return float(int(city_code) / 2500)


def f4(payment: str) -> int:
    if payment == "Pay Later":
        return 0
    else:
        return 1


def f5(hotel_type: str, map_dict: dict) -> float:
    return map_dict[hotel_type]


# def f5(hotel_type : str) -> float:
#     if hotel_type == "Hotel":
#         return 0.1
#     elif hotel_type == "Resort":
#         return 0.2
#     elif hotel_type == "Serviced Apartment":
#         return 0.3
#     elif hotel_type == "Apartment":
#         return 0.4
#     elif hotel_type == "Ryokan":
#         return 0.5
#     elif hotel_type == "Guest House / Bed & Breakfast":
#         return 0.6
#     elif hotel_type == "Bungalow":
#         return 0.7
#     elif hotel_type == "Hostel":
#         return 0.8
#     elif hotel_type == "UNKNOWN":
#         return 0.9
#     elif hotel_type == "Motel":
#         return 1
#     elif hotel_type == "Boat / Cruise":
#         return 0.15
#     elif hotel_type == "Resort Villa":
#         return 0.25
#     else:
#         return 0.3


def f6(hotel_star_rating: str) -> float:
    return float(int(hotel_star_rating) / 5)


# def f7(nationals_dict: dict, customer_nationality: str) -> float:
#     return nationals_dict[customer_nationality]


def f8(guest_is_not_the_customer: str) -> float:
    return float(guest_is_not_the_customer)


def f9(is_first_booking: str) -> float:
    return float(is_first_booking)


# def f10()


def f11(is_user_logged_in: str) -> float:
    if is_user_logged_in == "TRUE":
        return 0
    else:
        return 1


def f15(original_selling_amount: str) -> float:
    return float(float(original_selling_amount) / 5172)


def f16(request_airport: str) -> float:
    return float(request_airport)


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


##map customer_nationality to [0,1] inside a dict
def make_map_dict(df: pd.DataFrame, feature: str) -> dict:
    nationals_dict = {}
    counter = 0
    step = 1 / df[feature].nunique()
    for n in df[feature]:
        if n not in nationals_dict:
            nationals_dict[n] = counter
            counter += step
    return nationals_dict

def get_cancellation(features: pd.DataFrame):
    sum = []
    no_show = []
    for index, row in features.iterrows():
        a,b = f10(row.cancellation_policy_code, row.original_selling_amount, row.price_per_night)
        sum.append(a)
        no_show.append(b)
    return sum, no_show

def get_price_per_night(features: pd.DataFrame):
    price = []
    for index, row in features.iterrows():
        price.append( row.original_selling_amount / row.duration_nights)
    return price


def make_duration_nights(features: pd.DataFrame):
    a = []
    for index, row in features.iterrows():
        a.append((pd.to_datetime(row.checkout_date) - pd.to_datetime(row.checkin_date)).days)
    # amin, amax = min(a), max(a)
    #
    # for i, val in enumerate(a):
    #     a[i] = (val - amin) / (amax - amin)
    return a

def get_days_till_vac(features: pd.DataFrame):
    a = []
    for index, row in features.iterrows():
        a.append((pd.to_datetime(row.checkin_date) - pd.to_datetime(row.booking_datetime)).days)
    return a

def changeMatrix(features: pd.DataFrame) -> pd.DataFrame:
    new_features = pd.DataFrame()

    new_features["booking_hour"] = features["booking_datetime"].apply(f0)
    new_features["hotel_city_code"] = features["hotel_city_code"].apply(f_3)
    new_features["charge_option"] = features["charge_option"].apply(f4)
    new_features["accommadation_type_name"] = features["accommadation_type_name"].apply(f5, args = (make_map_dict(features, "accommadation_type_name"),))
    new_features["hotel_star_rating"] = features["hotel_star_rating"].apply(f6)
    new_features["customer_nationality"] = features["customer_nationality"].apply(f5, args = (make_map_dict(features, "customer_nationality"),))
    new_features["guest_is_not_the_customer"] = features["guest_is_not_the_customer"].apply(f8)
    new_features["is_first_booking"] = features["is_first_booking"].apply(f9)
    new_features["days_till_checkin"] = get_days_till_vac(features)
    new_features["is_user_logged_in"] = features["is_user_logged_in"].apply(f11)
    new_features["original_payment_method"] = features["original_payment_method"].apply(f5, args=(make_map_dict(features, "original_payment_method"),))
    new_features["no_of_adults"]= features["no_of_adults"].apply(f16)
    new_features["no_of_children"] = features["no_of_children"].apply(f16)
    new_features["original_selling_amount"] = features["original_selling_amount"]
    new_features["duration_nights"] = make_duration_nights(features)
    new_features["price_per_night"] = get_price_per_night(new_features)
    new_features["original_selling_amount"] = features["original_selling_amount"].apply(f5, args=(make_map_dict(features, "original_selling_amount"),))
    new_features["request_airport"] = features["request_airport"].apply(f16).fillna(0)
    new_features["cancellation_policy_code"] = features["cancellation_policy_code"]
    new_features["cancellation_sum"], new_features["cancellation_no_show"] =get_cancellation(new_features)
    del new_features["cancellation_policy_code"]
    del new_features["cancellation_no_show"]
    ##normalize:
    new_features["no_of_adults"] = (new_features["no_of_adults"] - new_features["no_of_adults"].min()) / (new_features["no_of_adults"].max() - new_features["no_of_adults"].min())
    new_features["no_of_children"] = (new_features["no_of_children"] - new_features["no_of_children"].min()) / (new_features["no_of_children"].max() - new_features["no_of_children"].min())
    new_features["duration_nights"] = (new_features["duration_nights"] - new_features["duration_nights"].min()) / (new_features["duration_nights"].max() - new_features["duration_nights"].min())
    new_features["days_till_checkin"] = (new_features["days_till_checkin"] - new_features["days_till_checkin"].min()) / (new_features["days_till_checkin"].max() - new_features["days_till_checkin"].min())
    return new_features


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
    # full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data = pd.read_csv(filename).drop_duplicates()
    # full_data["cancellation_datetime"] = full_data["cancellation_datetime"].fillna("2020-02-02")
    # full_data = full_data.dropna()
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
                          "request_airport"]]
    new_features = changeMatrix(features)
    labels = None
    if test:
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
