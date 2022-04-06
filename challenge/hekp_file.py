from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import re
import numpy as np
import pandas as pd

def make_condition_to_sum(cond: str, full_price: float, night_price: float) -> float:
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


if __name__ == '__main__':
    print(f10("130D1N_4N", 5000, 1000))