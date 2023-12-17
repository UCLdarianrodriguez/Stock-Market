""" Module for plotting
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
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
    """
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

def boxplot_outliers (data,name:str):

    """ plot outliers
    """

    data = data.iloc[:,1:]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    scaled_data = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame (optional)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # Creating a box plot for each column in the DataFrame
    plt.figure(figsize=(30,18))
    #df.boxplot()
    sns.boxplot(scaled_df,fill=False)
    #plt.boxplot(df)

    # Adding a title and labels
    plt.title('Box Plot of Columns')
    plt.xlabel('Features')
    plt.ylabel('Values')

    # Define the path where you want to save the plot
    folder_path = ("./figures" )

    plt.tight_layout()
    plt.savefig(f"{folder_path}/{name}.png")
