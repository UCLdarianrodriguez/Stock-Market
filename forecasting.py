""" train two models with the aim of forecasting the stock prices"""


import numpy as np
from numpy.typing import ArrayLike

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.stats import skew


def plot_series(ax: plt.axes, series: ArrayLike, params: dict, label: str) -> plt.Axes:
    """
    Plot a time series on a given matplotlib axis.

    Args:
        ax (plt.Axes): The matplotlib axis on which to plot the series.
        series (array-like): The time series data to be plotted.
        params (dict): A dictionary containing plot parameters.
            - 'time' (array-like): The time values corresponding to the series data.
            - 'title' (str): The title for the plot.
        label (str): The label for the series in the legend.

    Returns:
        (plt.Axes): matplotlib  axes objects
    """

    time = params["time"]
    ax.plot(time, series, label=label)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(params["title"])

    # Adding the legend
    ax.legend()

    plt.grid(True)

    return ax


def calculate_mape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between actual and forecast values.

    Args:
        actual (array-like): The actual values.
        forecast (array-like): The forecasted values.

    Returns:
        float: The MAPE value.
    """
    return np.mean(np.abs((actual - forecast) / actual)) * 100


def naive_forecaster(train: pd.DataFrame, test: pd.DataFrame):
    """
    Perform naive forecasting on time series data and
    generates a plot comparing the real and naive forecasted values

    Args:
        train (pd.DataFrame): Training dataset containing 'date' and 'close' columns.
        test (pd.DataFrame): Test dataset containing 'date' and 'close' columns.

    """

    # Separate variables
    y_train = train["close"]
    y_test = test["close"]

    # Shift the data to get the last value
    naive_forecast = list(y_test.shift(1))
    naive_forecast[0] = y_train.iloc[-1]

    parameters = {
        "title": "Naive Forecasting",
        "time": test["date"],
    }

    _, ax = plt.subplots(figsize=(10, 4))

    # Plot results
    ax = plot_series(ax, y_test, parameters, "Real")
    ax = plot_series(ax, naive_forecast, parameters, "Naive")

    # calculate mape and rsme
    mape = calculate_mape(np.array(y_test), np.array(naive_forecast))
    rmse = np.sqrt(mean_squared_error(np.array(y_test), np.array(naive_forecast)))

    print(f"Naive Forecasting MAPE is {mape:.2f}%")
    print(f"Naive Forecasting RMSE is {rmse:.2f}")

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/naive forecasting.png")
    plt.close()


def preprocess_data(
    data: np.array,
    split_val: int,
    split_test: int,
    window: int,
    enable_pca: bool = False
):
    """
    Preprocess the given time series data for LSTM model training.

    Args:
        data (numpy.ndarray): The time series data.
        split_val (int): Index to split the data into training and validation sets.
        split_test (int): Index to split the data into validation and test sets.
        window (int): Size of the window for creating sequences in the data.
        enable_pca (bool): Flag to enable (PCA) for dimensionality reduction.

    Returns:
        tuple: A tuple training, validation, and testing, along with the scaler.
    """

    # Splitting data
    scaled_train = data[:split_val]
    scaled_val = data[split_val:split_test]
    scaled_test = data[split_test:]

    print(scaled_train.shape)
    

    if enable_pca:
        # Run PCA and transform features
        pca = PCA(n_components=5)
        train_pca = pca.fit_transform(scaled_train[:, 1:])
        scaled_train = np.concatenate(
            (scaled_train[:, 0].reshape(-1, 1), train_pca), axis=1
        )
        test_scaled = pca.transform(scaled_test[:, 1:])
        scaled_test = np.concatenate(
            (scaled_test[:, 0].reshape(-1, 1), test_scaled), axis=1
        )
        val_pca = pca.transform(scaled_val[:, 1:])
        scaled_val = np.concatenate((scaled_val[:, 0].reshape(-1, 1), val_pca), axis=1)

        # Retrieve the explained variance ratios from the fitted model
        explained_variance = pca.explained_variance_ratio_
        print("PCA variance", explained_variance)
    


    # Separate the data
    x_train = scaled_train.copy()
    x_test = scaled_test.copy()
    y_train = scaled_train[:, 0]
    y_test = scaled_test[:, 0]
    x_val = scaled_val
    y_val = scaled_val[:, 0]

    # Concatenate data for slicing in the windowing
    x_concat = np.concatenate((x_train, x_val, x_test), axis=0)
    y_concat = np.concatenate((y_train, y_val, y_test), axis=0)

    # Prepare data for LSTM model
    xtrain, ytrain = [], []
    xtest, ytest = [], []
    xval, yval = [], []

    # creating windows with the data for training
    for i in range(window, split_val):
        xtrain.append(x_train[i - window : i, : x_train.shape[1]])
        ytrain.append(y_train[i])

    for i in range(split_val, split_test):
        xval.append(x_concat[i - window : i, : x_test.shape[1]])
        yval.append(y_concat[i])

    for i in range(split_test, len(x_concat)):
        xtest.append(x_concat[i - window : i, : x_test.shape[1]])
        ytest.append(y_concat[i])

    # convert everything to np.array
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xval, yval = np.array(xval), np.array(yval)
    xtest, ytest = np.array(xtest), np.array(ytest)

    if not enable_pca: pca = False

    return xtrain, ytrain, xtest, ytest, xval, yval, pca


def calculate_residuals(actual_values: ArrayLike, predicted_values: ArrayLike):
    """
    Calculate residuals and summary statistics of residuals.

    Args:
        actual_values (array-like): The actual values.
        predicted_values (array-like): The predicted values.
    """

    # Calculate residuals
    residuals = actual_values - predicted_values

    # Calculate mean, median, and skewness of residuals
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    skewness_residual = skew(residuals)  # Corrected function

    print(f"Mean of Residuals: {mean_residual}")
    print(f"Median of Residuals: {median_residual}")
    print(f"Skewness of Residuals: {skewness_residual}\n")


def prepare_data(new_data: pd.DataFrame, split_val: str, split_test: str) -> tuple:
    """
    Prepare and split time series data into training, validation, and test sets.

    Args:
        new_data (pd.DataFrame): The original time series data with a 'date' column.
        split_val (str): Date to split the data into training and validation sets.
        split_test (str): Date to split the data into validation and test sets.

    Returns:
        tuple: A tuple with preprocessed data, split indices, and dataframes for each set.
    """

    # getting the indexes of the split
    split_val_value = new_data[new_data["date"] >= split_val].index[0]
    split_test_value = new_data[new_data["date"] >= split_test].index[0]

    # dataframe for each set
    df_test = new_data[new_data["date"] >= split_test]
    df_validation = new_data[
        (new_data["date"] >= split_val) & (new_data["date"] < split_test)
    ]
    df_train = new_data[new_data["date"] < split_val]

    # setting  date as the index
    new_data = new_data.set_index("date", drop=True)

    # Preparing data for returning
    parameters = {
        "df_train": df_train,
        "df_val": df_validation,
        "df_test": df_test,
        "data": new_data,
    }

    return new_data, split_val_value, split_test_value, parameters

def preprocess_pca(data,split_val,split_test,window,pca):

    """
    Preprocesses the input data using Principal Component Analysis (PCA) and prepares it for LSTM model training.

    Args:
        data (numpy.ndarray): Input data matrix.
        split_val (int): Index for splitting data into training and validation sets.
        split_test (int): Index for splitting data into validation and test sets.
        window (int): Size of the sliding window for creating sequences.
        pca (PCA): Principal Component Analysis model.

    Returns:
        xtrain (numpy.array): Input features for training the LSTM model.
        ytrain (numpy.array): Target values for training the LSTM model.
        xval (numpy.array): Input features for validating the LSTM model.
        yval (numpy.array): Target values for validating the LSTM model.
    """
    
    #Splitting data 
    scaled_train = data[:split_val]
    scaled_val = data[split_val : split_test]
    scaled_test = data[split_test:]

    # PCA
    #pca = PCA(n_components=5)
    scaled_train = pca.fit_transform(scaled_train)
    print(scaled_test[:, 1:].shape)
    scaled_test = pca.transform(scaled_test)
    scaled_val = pca.transform(scaled_val)

    # Display the shapes of the original and final arrays
    print("Final Data Shape:", scaled_train.shape)

    # Retrieve the component loadings and explained variance ratios from the fitted model
    component_loadings = pca.components_
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    x_concat = np.concatenate((scaled_train, scaled_val, scaled_test), axis=0)

    # Prepare data for LSTM model
    xtrain, ytrain = [], []
    xval,yval = [], []

    # creating windows with the data for training
    for i in range(window, split_val):
        xtrain.append(x_concat[i - window : i, : x_concat.shape[1]])
        ytrain.append(x_concat[i])

    for i in range(split_val,split_test):
        xval.append(x_concat[i - window : i, : x_concat.shape[1]])
        yval.append(x_concat[i])


    xtrain,ytrain = np.array(xtrain),np.array(ytrain)
    xval,yval= np.array(xval),np.array(yval)

    return xtrain,ytrain,xval,yval

def fit_model_aux(xtrain,ytrain,xval,yval):

    """
    Build and train an LSTM model for time series prediction.

    Args:
        xtrain (array-like): Training input data with shape (samples, time steps, features).
        ytrain (array-like): Training output data.
        xval (array-like): Validation input data with shape (samples, time steps, features).
        yval (array-like): Validation output data.

    Returns:
        tuple: A tuple containing the training history and the trained LSTM model.
    """

    num_epochs = 60

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64,return_sequences = True,input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(xtrain.shape[2]))
    model.compile(optimizer='adam', loss = 'mean_squared_error')

    # Train the model
    history = model.fit(xtrain, ytrain, epochs=num_epochs, batch_size=16, verbose=0, validation_data=(xval,yval),shuffle=False)

    return history,model


def fit_model(
    xtrain: ArrayLike, ytrain: ArrayLike, xval: ArrayLike, yval: ArrayLike
) -> tuple:
    """
    Build and train an LSTM model for time series prediction.

    Args:
        xtrain (array-like): Training input data with shape (samples, time steps, features).
        ytrain (array-like): Training output data.
        xval (array-like): Validation input data with shape (samples, time steps, features).
        yval (array-like): Validation output data.

    Returns:
        tuple: A tuple containing the training history and the trained LSTM model.
    """
    # Set a seed for TensorFlow for reproducibility
    tf.random.set_seed(42)

    num_epochs = 30

    # Build LSTM model
    model = Sequential()
    model.add(
        LSTM(64, return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2]))
    )
    model.add(LSTM(64))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    history = model.fit(
        xtrain,
        ytrain,
        epochs=num_epochs,
        batch_size=16,
        verbose=0,
        validation_data=(xval, yval),
    )

    return history, model


def get_real_scale(
    data: np.array, num_features: int, scaler: StandardScaler
) -> np.array:
    """
    Reverse the scaling transformation to obtain the original scale of the data.

    Args:
        data (np.array): Scaled data.
        num_features (int): Number of features in the original data.
        scaler (StandardScaler): The scaler used for the transformation.

    Returns:
        numpy.array: Data in the original scale.
    """
    # setting the data into the format the scaler expects
    unscale = np.concatenate(
        (data, np.zeros((data.shape[0], num_features - 1))), axis=1
    )

    # reverse scaling
    unscale = scaler.inverse_transform(np.array(unscale))
    real_data = unscale[:, 0]

    return real_data


def plot_distribution(actual_values: np.array, predicted_values: np.array, name: str):
    """
    Create a joint plot to visualize the distribution of actual vs predicted values.

    Args:
        actual_values (np.array): Actual values.
        predicted_values (np.array): Predicted values.
        name (str): Name to include in the saved plot file.
    """

    plt.figure(figsize=(6, 6))

    # Create a joint plot
    sns.set(style="whitegrid")
    joint_plot = sns.jointplot(
        x=actual_values, y=predicted_values, kind="reg", color="skyblue"
    )

    # Customize the plot
    joint_plot.set_axis_labels("Actual Values", "Predicted Values", fontsize=12)
    joint_plot.fig.suptitle(
        "Joint Plot of Actual vs Predicted Values", y=1.02, fontsize=14
    )

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/jointplot {name}.png")
    plt.close()

def testing_predictions(x_concat,window,data_points,model):

    """
    Generate predictions using the provided model based on the input data.

    Args:
        x_concat (numpy.ndarray): Input data matrix containing historical features.
        window (int): Size of the sliding window used for prediction.
        data_points (int): Number of data points to predict.
        model (keras.Model): Trained machine learning model for making predictions.

    Returns:
        predictions_list (numpy.ndarray): Array containing the predicted values.
     """
 
    predictions_list = []
    num_features = x_concat.shape[1]
    x_concat = x_concat[-window:]

    for i in range(window,data_points+window):
        x = x_concat[i - window : i, : num_features]
        x = x.reshape(1,window,num_features)
        y = model.predict(x)
        predictions_list.append(y)
        x_concat = np.vstack((x_concat, y))

    predictions_list = np.array(predictions_list).reshape(-1,1)

    return predictions_list

def testing_predictions_aux(x_concat,window,data_points,models):
    """
    Generate predictions using the provided model based on the input data.

    Args:
        x_concat (numpy.ndarray): Input data matrix containing historical features.
        window (int): Size of the sliding window used for prediction.
        data_points (int): Number of data points to predict.
        model (keras.Model): Trained machine learning model for making predictions.

    Returns:
        predictions_list (numpy.ndarray): Array containing the predicted values.
     """

    predictions_list = []
    num_features = x_concat.shape[1]
    x_concat_aux = x_concat[:,1:][-window:]
    x_concat_price = x_concat[-window:]


    for i in range(window,data_points+window):
      
        x = x_concat_price[i - window : i, : num_features]
        x = x.reshape(1,window,num_features)
        price = models[0].predict(x)

        x = x_concat_aux[i - window : i, : num_features-1]
        x = x.reshape(1,window,num_features-1)
        aux = models[1].predict(x)

        joint = np.concatenate((price.reshape(-1, 1),aux),axis=1)

        predictions_list.append(price)

        x_concat_price = np.vstack((x_concat_price, joint))
        x_concat_aux = np.vstack((x_concat_aux, aux))

    predictions_list = np.array(predictions_list).reshape(-1,1)

    return predictions_list

def predict_plot(model, df_dict: dict, scaler, name: str):
    """
    Generate predictions using the provided LSTM model and plot the results.

    Args:
        model: The trained LSTM model.

        df_dict (dict): A dictionary containing DataFrames:
            - 'df_train': DataFrame for the training set
            - 'df_val': DataFrame for the validation set
            - 'df_test': DataFrame for the test set
            - 'xtest': test set with windows created
            - 'xval': validation set with windows created

        scaler: The scaler used for data normalization.
        name (str): Name to include in the saved plot files.
    """

    df_train = df_dict["df_train"]
    df_validation = df_dict["df_val"]
    df_test = df_dict["df_test"]
    xval = df_dict["xval"]
    test_predict = df_dict["prediction"]

    folder_path = "./figures"

    # predict data using test and validation set
    #test_predict = model.predict(xtest)
    val_predict = model.predict(xval)

    # count features, minus one to rest the date column
    num_features = df_train.shape[1] - 1

    #  format unscaled data
    y_test_unscaled = np.array(df_test["close"])
    y_val_unscaled = np.array(df_validation["close"])

    # convert prediction to original scale
    prediction_test = get_real_scale(test_predict, num_features, scaler)
    prediction_val = get_real_scale(val_predict, num_features, scaler)

    # compute metrics
    mape = calculate_mape(np.array(df_test["close"]), np.array(prediction_test))
    rmse = np.sqrt(
        mean_squared_error(np.array(y_test_unscaled), np.array(prediction_test))
    )

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"LSTM MAPE is {mape:.2f}%")

    # Plot actual and predicted values for the test set
    parameters = {"title": "LSTM Forecasting Test Set", "time": df_test["date"]}

    _, ax = plt.subplots(figsize=(10, 4))

    # Plot results
    ax = plot_series(ax, df_test["close"], parameters, "Real values")
    ax = plot_series(ax, prediction_test, parameters, "LSTM prediction")

    plt.savefig(f"{folder_path}/LSTM prediction {name}.png")
    plt.close()

    # Plot the whole dataset to compare the test set at the end of the period
    _, ax = plt.subplots(figsize=(10, 4))

    # plot the prediction and real values on the same axis
    ax.plot(df_train["date"], df_train["close"], label="Train")
    ax.plot(df_validation["date"], y_val_unscaled, label="Validation")
    ax.plot(df_test["date"], y_test_unscaled, label="Test")
    ax.plot(df_validation["date"], prediction_val, color="purple")
    ax.plot(df_test["date"], prediction_test, label="Prediction", color="purple")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("LSTM Forecasting")

    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder_path}/LSTM prediction overview {name}.png")
    plt.close()

    # compute residuals
    calculate_residuals(y_test_unscaled, prediction_test)
    plot_distribution(y_test_unscaled, prediction_test, name)


def forecasting_models(
    dataset: pd.DataFrame, train_set: pd.DataFrame, test_set: pd.DataFrame
):
    """
    Perform forecasting using univariate and multivariate models.

    Args:
        dataset (pd.DataFrame): The original dataset.
        train_set (pd.DataFrame): Training set for model training.
        test_set (pd.DataFrame): Testing set for evaluating model performance.
    """
    # ensure close in the 2nd column
    column_to_move = 'close'
    new_position = 1

    new_order = list(dataset.columns)
    new_order.insert(new_position, new_order.pop(new_order.index(column_to_move)))
    dataset = dataset[new_order]

    # Baseline model
    naive_forecaster(train_set, test_set)

    split_val = pd.to_datetime("2023-03-01")
    split_test = pd.to_datetime("2023-04-01")
    window_size = 4

    # Model 1 variable
    new_data = dataset.copy()

    # prepare data to train the model
    new_data, split_val_value, split_test_value, data_dict = prepare_data(
        new_data, split_val, split_test
    )

    # Scale data and convert to np ignore date column
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(data_dict['df_train'].iloc[:,1:])
    scaled_test = scaler.transform(data_dict['df_test'].iloc[:,1:])
    scaled_val = scaler.transform(data_dict['df_val'].iloc[:,1:])

    # joint in np array
    data_concat = np.concatenate((scaled_train, scaled_val, scaled_test), axis=0)


    xtrain, ytrain, xtest, _, xval, yval,_ = preprocess_data(
        np.array(data_concat[:,0].reshape(-1,1)),
        split_val_value,
        split_test_value,
        window=window_size,
    )

    # Run the model
    print("RUNNNG MODEL 1 - UNIVARIATE")
    _, model = fit_model(xtrain, ytrain, xval, yval)

    data_dict["xtest"] = xtest
    data_dict["xval"] = xval

    data_points = len(new_data) - split_test_value

    # getting recursively predictions
    prediction = testing_predictions(scaled_val[:,0].reshape(-1,1),window_size,data_points,model)

    # This predictions are scaled
    data_dict["prediction"] = prediction

    # predict with model and plot
    predict_plot(model, data_dict, scaler, "univariate")

    #################################################################### 
    # Run model 2

    print("RUNNING MODEL 2 - MULTIVARIATE")

    # only used for technical indicators eliminate 1st row
    data_concat = data_concat [1:,:]
    split_val_value = split_val_value -1 
    split_test_value = split_test_value -1


    # Run with pca
    xtrain, ytrain, xtest, _, xval, yval,pca = preprocess_data(
        data_concat, split_val_value, split_test_value, enable_pca=True, window=window_size
    )

    data_dict["xval"] = xval

    # Run the model for price in multivariate
    _, model_mult = fit_model(xtrain, ytrain, xval, yval)

    # ignore the close column to train the model
    xtrain,ytrain,xval,yval = preprocess_pca(data_concat[:,1:],split_val_value,split_test_value,window_size,pca)
    _,model_aux = fit_model_aux(xtrain,ytrain,xval,yval)

    # Predicting recursively
    data_points = len(data_concat) - split_test_value

    # Reduce dimensionality with PCA
    aux_tran = pca.transform(scaled_val[:,1:])
    x_concat = np.concatenate((scaled_val[:,0].reshape(-1, 1),aux_tran),axis=1)

    # getting recursively predictions
    prediction = testing_predictions_aux(x_concat,window_size,data_points,models = [model_mult,model_aux])

    # This predictions are scaled
    data_dict["prediction"] = prediction

    # predict with model multivariable and plot
    predict_plot(model_mult, data_dict, scaler, "multivariate")
