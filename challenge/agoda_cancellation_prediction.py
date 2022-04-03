from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator
from typing import NoReturn
import numpy as np
import pandas as pd

fields_who_cant_be_null = ['h_booking_id', 'booking_datetime', 'checkin_date', 'checkout_date', 'hotel_id',
                           "guest_is_not_the_customer", "original_selling_amount", "is_user_logged_in",
                           'charge_option', 'h_customer_id', 'accommadation_type_name']
# non_valid_cancellation_codes = ['0D0N', 'UNKNOWN']
non_valid_cancellation_codes = ['UNKNOWN']
# fields_who_cant_be_null = ['h_booking_id', 'booking_datetime', 'checkin_date', 'checkout_date', 'hotel_id',
#                            "guest_is_not_the_customer", "original_selling_amount", "is_user_logged_in",
#                            'charge_option', 'h_customer_id', 'accommadation_type_name', 'request_nonesmoke',
#                            'request_latecheckin', 'request_highfloor', 'request_largebed', 'request_twinbeds',
#                            'request_airport', 'request_earlycheckin']

# filter_list = ["hotel_id", "hotel_star_rating", "charge-option", "is_user_logged_in",
#                "request_largebed", "guest_is_not_the_customer"]
filter_list = ['charge_option_1', 'charge_option_2', 'hotel_star_rating', 'is_user_logged_in', 'is_first_booking']
dates_fields_to_format = ['booking_datetime', 'checkin_date', 'checkout_date', 'cancellation_datetime',
                          'hotel_live_date']

train_booking = [pd.to_datetime('2017-07-02').date(), pd.to_datetime('2018-09-30').date()]
train_checkout = [pd.to_datetime('2018-07-02').date(), pd.to_datetime('2018-09-30').date()]
test_booking = [pd.to_datetime('2018-11-01').date(), pd.to_datetime('2018-11-30').date()]
test_checkin = [pd.to_datetime('2018-12-15').date()]
test_cancellation = [pd.to_datetime('2018-12-07').date(), pd.to_datetime('2018-12-13').date()]


def check_validation(df: pd.DataFrame):
    df = df[df['booking_datetime'] <= df['checkin_date']]
    df = df[df['checkin_date'] < df['checkout_date']]
    df = df[df['hotel_live_date'] <= df['booking_datetime']]
    if 'cancellation_datetime' in df.columns:
        df = df.drop(
            df[df['cancellation_datetime'] >= df['checkout_date']].index)
    return df


def refactor_by_challenge_request(df: pd.DataFrame, fild_factor_1, fild_factor_2, fild_1_dates, fild_2_dates):
    df = df[pd.to_datetime(df[fild_factor_1]).dt.date.between(fild_1_dates[0], fild_1_dates[1])]
    if len(fild_2_dates) == 2:
        df = df[pd.to_datetime(df[fild_factor_2]).dt.date.between(fild_2_dates[0], fild_2_dates[1])]
    else:
        df = df[pd.to_datetime(df[fild_factor_2]).dt.date >= fild_2_dates[0]]
    return df


def load_data(filename: str, test_train):
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
    data = pd.read_csv(filename)
    # print("uniqe id: ", len(np.unique(full_data['cancellation_policy_code'])))
    if test_train == 'train':
        data = data.dropna(subset=fields_who_cant_be_null).drop_duplicates()
        data = check_validation(data)
    data["is_user_logged_in"] = np.where(data["is_user_logged_in"], 1, 0)
    data["is_first_booking"] = np.where(data["is_first_booking"], 1, 0)
    data["charge_option_1"] = np.where(data["charge_option"] == 'Pay Later', 1, 0)
    data["charge_option_2"] = np.where(data["charge_option"] == 'Pay Now', 1, 0)

    for field in dates_fields_to_format:
        format_datetime_to_date(data, field)
    if test_train == 'train':
        data["cancellation_datetime"] = np.where(data["cancellation_datetime"].isna(), 0, 1)
        data_after_filtration = refactor_by_challenge_request(data, 'checkout_date', 'booking_datetime',
                                                              train_checkout,
                                                              train_booking)
        features = data_after_filtration[filter_list]
        labels = data_after_filtration["cancellation_datetime"]
        return features, labels
    else:
        # data = data[data['accommadation_type_name'] == "Hotel"]
        # data = data[data['h_customer_id'] >= 0]
        # data = data[~data['cancellation_policy_code'].isin(non_valid_cancellation_codes)]
        # data = refactor_by_challenge_request(data, 'booking_datetime', 'checkin_date',
        #                                                       test_booking, test_checkin)

        # data = data_after_filtration[data['checkout_date'] >= test_cancellation[0]]
        # data.to_csv("Test data.csv", encoding='utf-8')
        return data[filter_list]


def format_datetime_to_date(df: pd.DataFrame, field: str) -> NoReturn:
    if field in df.columns:
        df[field] = pd.to_datetime(df[field]).dt.date


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
    x_p = estimator.predict(X)
    pd.DataFrame(x_p, columns=["predicted_values"]).to_csv(filename, index=False)
    # pd.DataFrame(y.values.ravel(), columns=["real_value"]).to_csv("2.csv", index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv", 'train')
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    # Fit model over data
    estimator = AgodaCancellationEstimator()
    estimator.fit(train_X.to_numpy(), train_y.values.ravel())
    # Store model predictions over test set
    test_file = '../datasets/test_set_week_1.csv'
    test_df = load_data(test_file, 'test')
    id1 = '208848317'
    id2 = '318189982'
    id3 = '315110833'
    csv_name = id1 + "_" + id2 + "_" + id3 + ".csv"
    evaluate_and_export(estimator, test_df.to_numpy(), csv_name)
    estimator.predict(train_X.values)
    print("Loss of model: ", estimator.loss(train_X.values, train_y.values))
    print("Test size: ", test_df.to_numpy().shape[0])
    print("the end!!!")
