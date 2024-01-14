""" This module replaces and analyze the missing values
"""

from pymongo.collection import Collection
import pandas as pd
import matplotlib.pyplot as plt
from store import connect_collections
from visualization import lineplot
from visualization import mark_data
from visualization import boxplot_outliers
import numpy as np
from numpy.typing import ArrayLike


def read_db(stock_data: Collection, aux_data: Collection) -> pd.DataFrame:
    """
    Read the information stored in Mongo database

    Args:
        stock_data (Collection): collection with stock prices data
        aux_data (Collection): collection with auxiliary data

    Returns:
        (pd.Dataframe): dataframe with stock prices
        (pd.Dataframe): dataframe with auxiliary data
    """

    # excluding the _id field
    projection = {"_id": 0}

    # Query the database
    df_economic_data = pd.DataFrame(aux_data.find({}, projection))
    df_stock = pd.DataFrame(stock_data.find({}, projection))

    return df_stock, df_economic_data


def missing_percentange(data: pd.DataFrame):
    """
    Calculate the missing values percentage and display it

    Args:
        data (pd.DataFrame): data to caculate missing percentage
    """

    # calculate percentage of missing values
    miss_percentage = data.isna().mean() * 100

    # convert to two decimal places
    rounded_values = miss_percentage.round(2)

    # create a table for reporting
    missing_df = pd.DataFrame(
        {"Feature": miss_percentage.index, "Missing %": rounded_values.values}
    )

    missing_df["Missing %"] = missing_df["Missing %"].apply(
        lambda x: "{:.2f}%".format(x)
    )
    print(missing_df)


def impute_missing(data: pd.DataFrame) -> (list, pd.DataFrame):
    """
    Impute missing data using forward filling

    Args:
        data (pd.Dataframe): dataframe with the data

    Returns:
        (list): list of indexes to impute
        (pd.Dataframe): dataframe with missing values imputed
    """

    # list of indexes to impute
    matrix_index = data.isna()

    # Forward-fill missing values
    filled_df = data.ffill(axis=0)

    return matrix_index, filled_df


def get_range(df_data: pd.DataFrame):
    """
    Evaluate the range values of each feature

    Args:
        data (pd.Dataframe): dataframe with the data

    """
    # exclude date column
    df_data = df_data.iloc[:, 1:]

    # calculate min and max values per column
    min_values = df_data.min(axis=0)
    max_values = df_data.max(axis=0)
    table = pd.DataFrame({"Min": min_values, "Max": max_values})
    print(table)

def iqr_detect(x: ArrayLike, threshold: float=1.5) -> ArrayLike:
    """
    Detects outliers using the interquantile range method and returns the indices of the outliers.

    Args:
        x (ArrayLike): data to be checked for outliers
        threshold (float): threshold for the interquantile range method

    Returns:
        (ArrayLike): indices of the outliers
    """

    # calculate the Q1 and Q3 quantiles
    x = np.array(x)
    Q1 = np.quantile(x, 0.25)
    Q3 = np.quantile(x, 0.75)

    #Whiskers min and max
    IQR = Q3-Q1
    IQR_min = Q1 - threshold*IQR
    IQR_max = Q3 + threshold*IQR

    #Extract outliers indexes
    indexes = np.where((x>IQR_max)|(x<IQR_min))

    return indexes[0]


def train_test_split(data, test_start_date: str) -> pd.DataFrame:
    """
    split dataset into train and validation sets

    Args:
        data (pd.Dataframe): dataframe with the data
         test_start_date: first day of test set
    Returns:
        (pd.Dataframe): train set
        (pd.Dataframe): test set
    """
    # get row index of the split date
    split_index = data[data["date"] >= test_start_date].index[0]

    # Split data into train and test
    df_train = data.loc[: split_index - 1].reset_index(drop=True)
    df_test = data.loc[split_index:].reset_index(drop=True)

    return df_train, df_test


def missing_figures(data: pd.DataFrame, indexes):
    """
    plot the missing data imputed

    Args:
        data (pd.Dataframe): dataframe with the data
        indexes (list): indexes of missing data
    Returns:

    """

    time_series = data.copy()

    # Define the path where you want to save the plot
    folder_path = ("./figures" )

    # Selected feature
    features = ["T5YIE", "INFECTDISEMVTRACKD", "VIXCLS", "TREASURY_YIELD"]

    fig, ax = plt.subplots(len(features), 1, figsize=(30, 10))

    for i, feature in enumerate(features):
        # Plot configuration
        colors = "green"
        legend = [feature]

        # Data to plot
        x = time_series["date"].dt.date
        y = time_series[feature]

        # Call the plots
        ax[i] = lineplot(
            x, y, f"{feature} Missing values", "Date", "value", legend, colors, ax[i]
        )
        ax[i] = mark_data(ax[i], x, y, indexes[indexes[feature]].index)

    # Save the plot to the specified folder with a file name
    plt.tight_layout()
    plt.savefig(f"{folder_path}/missing_values.png")

def outliers_figure(data: pd.DataFrame, name: str):
    """
    box plot and time plot for outliers

    Args:
        data (pd.Dataframe): dataframe with the data
        name (str): name of the photo
    Returns:

    """

    time_series = data.copy()

    # Define the path where you want to save the plot
    folder_path = ("./figures")

    # Selected feature
    features = data.iloc[:,1:].columns

    fig, ax = plt.subplots(len(features), 1, figsize=(40, 40))

    for i, feature in enumerate(features):

        # Get outliers indexes
        indexes = iqr_detect(data[feature])
 
        # Plot configuration
        colors = "green"
        legend = [feature]

        # Data to plot
        x = time_series["date"].dt.date
        y = time_series[feature]

        # Call the plots
        ax[i] = lineplot(
            x, y, f"{feature} outliers", "Date", "value", legend, colors, ax[i]
        )
        ax[i] = mark_data(ax[i], x, y, indexes)

    
    # Adjust space between subplots
    plt.tight_layout()

    # Save the plot to the specified folder with a file name
    plt.savefig(f"{folder_path}/{name}.png")  


def clean_data(test_split:str) -> pd.DataFrame:
    """
    read, clean, format the data stored in Mongo and split it into train and test set

    Args:
        test_split(str): first day of test set

    Returns:
        (pd.Dataframe): data merged and cleaned
        (pd.Dataframe): train set
        (pd.Dataframe): test set

    """
    # Connect with the collections
    stock_collection, aux_collection = connect_collections()

    # Read the information stored
    df_stock_prices, df_aux_economic_data = read_db(stock_collection, aux_collection)

    # Calculate missing values percentage for aux data
    # missing_percentange(df_aux_economic_data)

    # Calculate missing values percentage for stock data
    # missing_percentange(df_stock_prices)

    # Only extract the dates contained in stock dataframe
    df_aux_economic_data = df_aux_economic_data[
        df_aux_economic_data["date"].isin(df_stock_prices["date"])
    ]

    # Calculate missing values percentage for aux data
    missing_percentange(df_aux_economic_data)

    # Impute missing data
    indexes, df_aux_imputed = impute_missing(df_aux_economic_data)

    # Analyze the range of values
    get_range(df_stock_prices)
    get_range(df_aux_economic_data)

    # Visualize time series before and after imputation
    missing_figures(df_aux_imputed, indexes)

    # outliers with boxplot
    boxplot_outliers(df_aux_imputed,"outliers in Auxiliary")
    boxplot_outliers(df_stock_prices,"outliers in Stocks")

    df_aux_imputed = df_aux_imputed.reset_index(drop=True)

    # outliers in time domain
    outliers_figure(df_stock_prices,"outliers in Stocks time domain")
    outliers_figure(df_aux_imputed,"outliers in Aux time domain")


    # Merge dataset
    pd_concat = pd.merge(df_stock_prices, df_aux_imputed, on="date", how="inner").reset_index(drop=True)

    train_set,test_set = train_test_split(pd_concat,test_split)

    return pd_concat,train_set,test_set
