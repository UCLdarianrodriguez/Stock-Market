""" Module for plotting
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import pandas as pd


def lineplot(
    x: ArrayLike,
    ys: ArrayLike,
    title: str,
    x_label: str,
    y_label: str,
    legend: List[str],
    plt_color: str,
    ax: plt.Axes,
) -> plt.Axes:
    """cls
    Creates a line plot

    Args:
        x (ArrayLike): value of the data on the x-axis
        ys (List[ArrayLike]): list of values of the data on the y-axis
        title (str): title of the plot
        x_label (str): x-axis label
        y_label (str): y-axis label
        legend (List[str]): list of legend labels
        plt_color(list[str]): line color list of the plot

    Returns:
        (plt.Axes): matplotlib  axes objects
    """

    ax.plot(x, ys, color=plt_color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # configuring x axis ticks
    ax.set_xticks(ticks=x[::30], labels=x[::30], rotation=60)

    # Adding the legend
    ax.legend(legend)

    return ax


def mark_data(
    ax: plt.Axes, x: ArrayLike, y: ArrayLike, outliers_idx: ArrayLike
) -> plt.Axes:
    """
    Plots red circles in a certain line plot.

    Args:
        ax (plt.Axis): the older axis to plot on that contains the original line plot
        x (ArrayLike): the x-data for the line plot
        y (ArrayLike): the y-data for the line plot
        data_idx (ArrayLike): indices of the data to mark
    """

    y_out = y[outliers_idx]
    x_out = x[outliers_idx]

    # Plot outliers
    ax.scatter(x_out, y_out, color="red", marker="o", alpha=0.8)

    return ax


def boxplot_outliers(data, name: str):
    """
    plot outliers

    Args:
        name(str): filename
    """

    data = data.iloc[:, 1:]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    scaled_data = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame (optional)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # Creating a box plot for each column in the DataFrame
    plt.figure(figsize=(30, 18))

    sns.boxplot(scaled_df, fill=False)

    # Adding a title and labels
    plt.title("Box Plot of Columns")
    plt.xlabel("Features")
    plt.ylabel("Values")

    # Define the path where you want to save the plot
    folder_path = "./figures"

    plt.tight_layout()
    plt.savefig(f"{folder_path}/{name}.png")


def stacionarity_test(data: ArrayLike):
    """
    Perform the Augmented Dickey-Fuller test for stationarity on a time series, where
    the null hypothesis of this test is that the series is non-stationary.

    Args:
        data (ArrayLike): Time series data to be tested for stationarity.

    Prints:
    - Test Statistic: The test statistic of the Augmented Dickey-Fuller test.
    - p-value: The p-value associated with the test.
    - Whether to reject the null hypothesis based on the p-value (0.05 significance level).
    - Critical Values: The critical values at different confidence levels.

    Returns:
    None
    """

    result = adfuller(data)

    # Extract and print the test statistic and p-value
    test_statistic = result[0]
    p_value = result[1]

    print(f"Test Statistic: {test_statistic}")
    print(f"p-value: {p_value}")

    # Check the p-value against a significance level (0.05)
    if p_value <= 0.05:
        print("Reject the null hypothesis; the data is stationary.")
    else:
        print("Fail to reject the null hypothesis; the data is non-stationary.")

    print("Critical Values:")
    for thres, adf_stat in result[4].items():
        print(f"{thres}: {adf_stat:.2f}")

    print("\n")


def correlation_matrix(data: pd.DataFrame, name: str):
    """
    Plots correlation matrix for a given dataset.

    Args:
        data (pd.DataFrame): The dataset used.
        name(str): filename
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(data.corr(), annot=True, cbar=False, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Features Correlation Matrix")
    plt.tight_layout()

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{name}.png")


def plot_month_year(data: pd.DataFrame, name: str):
    """
    Plot the average volume over months using a bar plot and a heatmap.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing a 'date' column and a 'volume' column.
        name (str): Name of the plot file to be saved (filename)
    """

    # extract the year and month components from the index and add them as new columns
    month_year = data.copy()
    month_year.loc[:, "year"] = data["date"].dt.year
    month_year.loc[:, "month"] = data["date"].dt.month

    figure, ax = plt.subplots(1, 2, figsize=(12, 4))

    # volumen bar plot
    month_year.groupby(["month"])["volume"].mean().plot(kind="bar", ax=ax[0])

    # group data for the heatmap
    month_year = month_year.groupby(["year", "month"])["volume"].mean().unstack()

    # Heatmap plot
    sns.heatmap(
        month_year, ax=ax[1], cmap="coolwarm", linestyle="dashed", linecolor="white"
    )

    ax[1].set_title("Volume over months Heatmap")
    ax[0].set_title("Volume over months Bar Plot")
    ax[0].set_ylabel("Volume")

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{name}.png")


def shaded_plot(x: ArrayLike, ys: List[ArrayLike], params) -> (plt.Figure, plt.Axes):
    """
    Creates a shaded are plot for each list of data in the given list of data, title and labels.

    Args:
        x (ArrayLike): value of the data on the x-axis
        ys (List[ArrayLike]): list of values of the data on the y-axis, in the order [mean,min,max]
        params(dict): A dictionary containing the following keys:
            - 'title' (str): Title of the plot
            - 'x_label' (str): Label for the X-axis
            - 'y_label' (str): Label for the Y-axis
            - 'opacity' (float): Opacity of the scatter points
            - 'name' (str): filename



    Returns:
        (plt.Figure, plt.Axes): matplotlib figure and axes objects
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    # Get the central plot
    ax = lineplot(
        x,
        ys[0],
        params["title"],
        params["x_label"],
        params["y_label"],
        plt_color="green",
        legend="",
        ax=ax,
    )

    # Fill the area between the curves
    ax.fill_between(x, ys[1], ys[2], color="green", alpha=params["opacity"])
    plt.tight_layout()

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{params['name']}.png")

    return fig, ax


def plot_closed_price(ax, x, y):
    """
    Plot the closing prices.

    Args:
        ax (matplotlib.axes): Axes object to plot on.
        x (array-like): X-axis values (dates).
        y (array-like): Y-axis values (closing prices).

    Returns:
        plt.Axes: matplotlib axes objects
    """

    # Plot close price
    ax.plot(x, y, label="Close Price")

    # Customize the plot
    ax.grid(True)
    ax.set_title("Closing Price")
    ax.set_ylabel("Price")
    ax.legend()

    return ax


def calculate_macd(data, name: str):
    """
    Calculate MACD (Moving Average Convergence Divergence) and plot the results.

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'date' column and a 'close' column.
        name(str): filename

    Returns:
        (pd.DataFrame): DataFrame with MACD Line, Signal Line, and MACD Histogram column
    """

    # Create an empty dataframe
    final_data = pd.DataFrame()

    # Calculate 12-day EMA
    ema_12 = data["close"].ewm(span=12, adjust=False).mean()

    # Calculate 26-day EMA
    ema_26 = data["close"].ewm(span=26, adjust=False).mean()

    # Calculate MACD Line
    final_data["MACD_Line"] = ema_12 - ema_26

    # Calculate 9-day Signal Line
    final_data["Signal_Line"] = final_data["MACD_Line"].ewm(span=9, adjust=False).mean()

    # Calculate MACD Histogram
    final_data["MACD_Histogram"] = final_data["MACD_Line"] - final_data["Signal_Line"]

    # Plotting MACD Line and Signal Line
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[1].plot(data["date"], final_data["MACD_Line"], label="MACD Line", color="blue")
    axs[1].plot(
        data["date"], final_data["Signal_Line"], label="Signal Line", color="orange"
    )

    # Plotting MACD Histogram with positive values in green and negative values in red
    axs[1].bar(
        data["date"],
        final_data["MACD_Histogram"],
        label="MACD Histogram",
        color=[
            "green" if value > 0 else "red" for value in final_data["MACD_Histogram"]
        ],
    )

    # Add horizontal line at y=0 for MACD Histogram
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1)

    # Adding closing price to the subplot
    plot_closed_price(axs[0], data["date"], data["close"])

    # Customize the plot
    axs[1].grid(True)
    axs[1].set_title("MACD Analysis")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("MACD Values")
    axs[1].legend()

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{name}.png")

    return final_data


def calculate_adx(data: pd.DataFrame, name: str, period: int = 14):
    """
    Calculate the Average Directional Index (ADX) along with +DI and -DI.

    Args:
        data (pd.DataFrame): Input DataFrame containing columns 'date', 'high', 'low', and 'close'.
        name(str): filename
        period (int, optional): The time period for calculating ADX, +DI, and -DI. Default is 14.

    Returns:
        (list): calculated ADX
    """

    # Calculate True Range (TR)
    data["TR"] = pd.DataFrame(
        [
            data["high"] - data["low"],
            abs(data["high"] - data["close"].shift()),
            abs(data["low"] - data["close"].shift()),
        ]
    ).max(axis=0)

    #  calculates the difference between consecutive high prices
    data["UpMove"] = data["high"].diff()

    # calculates the negative difference between consecutive low prices
    data["DownMove"] = -data["low"].diff()

    # Calculate +DM and -DM
    data["+DM"] = np.where(
        (data["UpMove"] > data["DownMove"]) & (data["UpMove"] > 0), data["UpMove"], 0
    )
    data["-DM"] = np.where(
        (data["DownMove"] > data["UpMove"]) & (data["DownMove"] > 0),
        data["DownMove"],
        0,
    )

    # Smoothed +DM and -DM
    data["Smoothed_+DM"] = data["+DM"].ewm(span=period, adjust=False).mean()
    data["Smoothed_-DM"] = data["-DM"].ewm(span=period, adjust=False).mean()

    # Calculate Directional Movement Index (DX)
    data["DX"] = (
        100
        * (
            abs(data["Smoothed_+DM"] - data["Smoothed_-DM"])
            / (data["Smoothed_+DM"] + data["Smoothed_-DM"])
        )
        .ewm(span=period, adjust=False)
        .mean()
    )

    # Calculate Average Directional Index (ADX)
    data["ADX"] = data["DX"].ewm(span=period, adjust=False).mean()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    plot_closed_price(axs[0], data["date"], data["close"])

    # Plot ADX
    axs[1].plot(data["date"], data["ADX"], label="ADX", color="purple")
    axs[1].axhline(
        20, color="orange", linestyle="--", linewidth=1, label="ADX Level 20"
    )
    axs[1].axhline(50, color="blue", linestyle="--", linewidth=1, label="ADX Level 50")
    axs[1].axhline(70, color="red", linestyle="--", linewidth=1, label="ADX Level 70")
    axs[1].legend()
    axs[1].set_title("Average Directional Index (ADX)")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("ADX Values")
    axs[1].grid(True)

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{name}.png")

    return data["ADX"]


def calculate_rsi(data: pd.DataFrame, name: str, period: int = 14):
    """
    Calculate the Relative Strength Index (RSI) for the close price

    Args:
        data (pd.DataFrame): Input DataFrame containing the time series data including close price
        name(str): filename
        period (int, optional): Time period for RSI calculation. Default is 14.

    Returns:
        (list):calculated RSI.
    """

    # Calculate price changes
    data["price_change"] = data["close"].diff()

    # Calculate gains and losses
    data["gain"] = np.where(data["price_change"] > 0, data["price_change"], 0)
    data["loss"] = np.where(data["price_change"] < 0, -data["price_change"], 0)

    # Calculate average gains and losses over the specified period
    avg_gain = data["gain"].rolling(window=period, min_periods=1).mean()
    avg_loss = data["loss"].rolling(window=period, min_periods=1).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    data["rsi"] = 100 - (100 / (1 + rs))

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    plot_closed_price(axs[0], data["date"], data["close"])

    # Plot RSI
    axs[1].plot(data["date"], data["rsi"], label="RSI", color="purple")
    axs[1].axhline(
        70, color="red", linestyle="--", linewidth=1, label="Overbought (70)"
    )
    axs[1].axhline(
        30, color="green", linestyle="--", linewidth=1, label="Oversold (30)"
    )
    axs[1].axhline(70, color="red", linestyle="--", linewidth=1, label="ADX Level 70")
    axs[1].legend()
    axs[1].set_title("Relative Strength Index (RSI)")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("RSI Values")
    axs[1].grid(True)

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{name}.png")

    return data["rsi"]


def calculate_obv(data: pd.DataFrame, name: str):
    """
    Calculate On-Balance Volume (OBV) for a given DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing time series data.
        name(str): filename

    Returns:
        (list):calculated OBV.
    """

    # Empty list to save the OBV values calculated
    obv_values = []

    for i in range(1, len(data)):
        # Check if the closing price increased
        if data["close"].iloc[i] > data["close"].iloc[i - 1]:
            # Calculate OBV for increasing price
            obv_values.append(
                data["volume"].iloc[i] + obv_values[-1]
                if obv_values
                else data["volume"].iloc[i]
            )

        # Check if the closing price decreased
        elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
            obv_values.append(
                obv_values[-1] - data["volume"].iloc[i]
                if obv_values
                else -data["volume"].iloc[i]
            )

        # Check if the closing price remained the same
        else:
            obv_values.append(obv_values[-1] if obv_values else 0)

    # Add the OBV values to the DataFrame where the first value is zero
    obv = [0] + obv_values

    # Plot OBV
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plot_closed_price(axs[0], data["date"], data["close"])

    axs[1].plot(data["date"], obv, label="On-Balance Volume (OBV)", color="purple")
    axs[1].set_title("On-Balance Volume (OBV)")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("OBV Values")
    axs[1].legend()
    axs[1].grid(True)

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/{name}.png")

    return obv
