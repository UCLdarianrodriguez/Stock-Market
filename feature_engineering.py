"""
 Modules for feature engineeering
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from visualization import (
    calculate_macd,
    calculate_adx,
    calculate_obv,
    calculate_rsi,
)

from clean import train_test_split

def create_features(data,columns_to_drop:list,split_day:str):
    """
    Eliminate redundant features and create technical indicator features

    Args:
        data (pd.DataFrame): Input DataFrame containing the time series data
        columns_to_drop(list): list of features to drop
        split_day: start day of the test set


    Return:
        (pd.DataFrame): training set
        (pd.DataFrame: test set
    """

    df_non_redundant = data.drop(columns=columns_to_drop)

    # Calculate financial indicators
    indicator_macd = calculate_macd(data.copy(), plot=False)
    indicator_adx = calculate_adx(data.copy(), period=14, plot=False)
    indicator_rsi = calculate_rsi(data.copy(), plot=False)
    indicator_obv = calculate_obv(data.copy(), plot=False)

    # Putting the technical indicator together
    df_technical = pd.DataFrame(
        {"obv": indicator_obv, "rsi": indicator_rsi, "adx": indicator_adx}
    )

    df_final_data = pd.concat([df_non_redundant,df_technical, indicator_macd], axis=1)

    train_set,test_set = train_test_split(df_final_data,split_day)

    return df_final_data,train_set,test_set

