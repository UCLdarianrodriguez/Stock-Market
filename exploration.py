"""
This module is designed for a thorough Exploratory Data Analysis (EDA) on financial data,
specifically focusing on indicators commonly used in technical analysis, 
discover relationships within the data and eliminate redundant features.

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from visualization import (
    stacionarity_test,
    correlation_matrix,
    plot_month_year,
    shaded_plot,
    calculate_macd,
    calculate_adx,
    calculate_obv,
    calculate_rsi,
    season_decomposition,
)


def eda(data: pd.DataFrame):
    """
    Perform  Exploratory Data Analysis on the data

    Args:
        data (pd.DataFrame): Input DataFrame containing the time series data

    Return:
        (list): features redundant

    """

    # statistic verification if the series closed price is non-stationary
    stacionarity_test(data["close"])

    # Removing this features for visualization purposes since is already known they are redundant
    columns_to_drop = ["unadjustedVolume", "adjClose"]
    df_non_redundant = data.drop(columns=columns_to_drop)

    # Pearson correlation coefficient  for Redundant Feature Removal, skip date column
    correlation_matrix(df_non_redundant.iloc[:, 1:], "Correlation all features")

    # Eliminate features with high correlation between them
    redundant_feat = [
        "vwap",
        "low",
        "open",
        "high",
        "changePercent",
        "changeOverTime",
        "DFF",
    ]
    df_non_redundant = df_non_redundant.drop(columns=redundant_feat)

    # Plotting the final correlation with selected features
    correlation_matrix(
        df_non_redundant.iloc[:, 1:], "Correlation features non redundant"
    )

    # Rename columns for visualization
    column_mapping = {
        "INFECTDISEMVTRACKD": "INFECTED",
        "TREASURY_YIELD": "YIELD",
    }

    df_renamed = df_non_redundant.rename(columns=column_mapping)

    # Generate grid of scatterplots for pairwise relationships in the dataset
    sns.pairplot(data=df_renamed, aspect=1, height=1)
    plt.tight_layout()

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/pairplot.png")

    # Explore relationships between the month and the year
    plot_month_year(df_non_redundant, "monthly_year")

    # Time plot over months

    # Create a new figure
    plt.figure(figsize=(20, 8))

    # Resample data to monthly
    monthly_mean = df_non_redundant.resample("M", on="date").mean()

    # Plot data
    monthly_mean.plot(
        subplots=True,
        layout=(3, 3),
        figsize=(15, 15),
        sharex=False,
        title="Stock Value Trend from 2019 - 2023 (Month)",
        rot=90,
    )
    plt.subplots_adjust(top=0.94, hspace=0.3)

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/monthly_mean.png")

    # Plot parameters
    parameters = {
        "title": "Closing Price Range",
        "x_label": "Date",
        "y_label": "Price",
        "opacity": 0.3,
        "name": "close price Shaded plot",
    }

    # get the data for the plot
    x = data["date"].dt.date
    y = data["close"]
    y_min = data["low"]
    y_max = data["high"]
    ys = [y, y_min, y_max]

    # Shaded Plot to explore price ranges over time
    shaded_plot(x, ys, params=parameters)

    # Calculate financial indicators
    indicator_macd = calculate_macd(data.copy(), name="MACD")
    indicator_adx = calculate_adx(data.copy(), period=14, name="ADX")
    indicator_rsi = calculate_rsi(data.copy(), name="RSI")
    indicator_obv = calculate_obv(data.copy(), name="OBV")

    # Putting the technical indicator together
    df_technical = pd.DataFrame(
        {"obv": indicator_obv, "rsi": indicator_rsi, "adx": indicator_adx}
    )
    df_technical = pd.concat([df_technical, indicator_macd], axis=1)

    # Analyze Correlation with technical indicators
    correlation_matrix(
        pd.concat([df_technical, df_non_redundant.iloc[:, 1:]], axis=1),
        "Correlation technical indicators",
    )

    season_decomposition(
        df_non_redundant["close"],
        df_non_redundant["date"].dt.date,
        365,
        "Seasonal Decomposition",
    )

    features_drop = redundant_feat + columns_to_drop

    return features_drop
