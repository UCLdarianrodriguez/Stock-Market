""" train two models with the aim of forecasting the stock prices"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings



def plot_series(ax,series,params,label):

    time = params['time']
    ax.plot(time, series,label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(params['title'])

    # Adding the legend
    ax.legend()

    plt.grid(True)

    return ax

calculate_mape = lambda actual, forecast: np.mean(np.abs((actual - forecast) / actual)) * 100

def naive_forecaster(train, test):

    # Separate variables
    x_train = train.drop('close',axis=1)
    x_test = test.drop('close',axis=1)
    y_train = train['close']
    y_test = test['close']

    # Shift the data to get the last value
    naive_forecast = list(y_test.shift(1))
    naive_forecast[0] = y_train.iloc[-1]

    parameters = {
        'title': "Naive Forecasting",
        'time': test['date'],
    }

    fig, ax = plt.subplots(figsize=(10,4))

    # Plot results
    ax = plot_series(ax,y_test,parameters,"Real")
    ax = plot_series(ax,naive_forecast,parameters,"Naive")

    mape = calculate_mape(np.array(y_test), np.array(naive_forecast))
    rmse = np.sqrt(mean_squared_error(np.array(y_test),np.array(naive_forecast)))

    print(f"Naive Forecasting MAPE is {mape:.2f}%")
    print(f"Naive Forecasting RMSE is {rmse:.2f}")

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/naive forecasting.png")
    plt.close()

def preprocess_data(data,split_val,split_test,window,enable_pca:bool=False):
    
    #Splitting data 
    data_train = data[:split_val]
    data_val = data[split_val : split_test]
    data_test = data[split_test:]

    #Scale data
    scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(data_train)
    scaled_test = scaler.transform(data_test)
    scaled_val = scaler.transform(data_val)


    if enable_pca:
        # PCA
        pca = PCA(n_components=5)
        train_pca = pca.fit_transform(scaled_train[:, 1:])
        scaled_train = np.concatenate((scaled_train[:, 0].reshape(-1, 1), train_pca), axis=1)
        test_scaled = pca.transform(scaled_test[:, 1:])
        scaled_test = np.concatenate((scaled_test[:, 0].reshape(-1, 1), test_scaled), axis=1)
        val_pca = pca.transform(scaled_val[:, 1:])
        scaled_val = np.concatenate((scaled_val[:, 0].reshape(-1, 1), val_pca), axis=1)

        # Retrieve the component loadings and explained variance ratios from the fitted model
        explained_variance = pca.explained_variance_ratio_
        print("PCA variance",explained_variance)


    # Separate
    x_train= scaled_train.copy()
    x_test= scaled_test .copy()
    y_train = scaled_train[:,0]
    y_test = scaled_test[:,0]
    x_val = scaled_val
    y_val = scaled_val[:,0]

    x_concat = np.concatenate((x_train, x_val, x_test), axis=0)
    y_concat = np.concatenate((y_train, y_val, y_test), axis=0)

    # Prepare data for LSTM model
    xtrain, ytrain = [], []
    xtest,ytest = [], []
    xval,yval = [], []

    # creating windows with the data for training
    for i in range(window, split_val):
        xtrain.append(x_train[i - window : i, : x_train.shape[1]])
        ytrain.append(y_train[i])

    for i in range(split_val,split_test):
        xval.append(x_concat[i - window : i, : x_test.shape[1]])
        yval.append(y_concat[i])

    for i in range(split_test,len(x_concat)):
        xtest.append(x_concat[i - window : i, : x_test.shape[1]])
        ytest.append(y_concat[i])


    xtrain,ytrain = np.array(xtrain),np.array(ytrain)
    xval,yval= np.array(xval),np.array(yval)
    xtest,ytest= np.array(xtest),np.array(ytest)

    return xtrain,ytrain,xtest,ytest,xval,yval,scaler

def calculate_residuals(actual_values,predicted_values):
    # Calculate residuals
    residuals = actual_values - predicted_values

    # Calculate mean, median, and skewness of residuals
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    skewness_residual = skew(residuals)  # Corrected function

    print(f"Mean of Residuals: {mean_residual}")
    print(f"Median of Residuals: {median_residual}")
    print(f"Skewness of Residuals: {skewness_residual}\n")

def prepare_data(new_data,split_val,split_test):

    # getting the indexes of the split
    split_val_value = new_data[new_data['date'] >= split_val].index[0]
    split_test_value = new_data[new_data['date'] >= split_test].index[0]

    # dataframe for each set
    df_test = new_data[new_data['date'] >= split_test]
    df_validation = new_data[(new_data['date'] >= split_val) & (new_data['date'] < split_test)]
    df_train = new_data[new_data['date'] < split_val]

    # setting  date as the index 
    new_data = new_data.set_index('date',drop=True)

    parameters = {
    'df_train': df_train,
    'df_val': df_validation,
    'df_test': df_test,
    'data': new_data
    }

    return new_data,split_val_value,split_test_value,parameters

def fit_model(xtrain,ytrain,xval,yval):

    num_epochs = 60

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64,return_sequences = True,input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss = 'mean_squared_error')

    # Train the model
    history = model.fit(xtrain, ytrain, epochs=num_epochs, batch_size=16, verbose=0, validation_data=(xval,yval))

    return history,model

def get_real_scale(data,num_features,scaler):

    unscale = np.concatenate((data, np.zeros((data.shape[0],num_features-1 ))), axis=1)
    unscale   = scaler.inverse_transform(np.array(unscale))
    real_data = unscale[:,0]
    return real_data

def plot_distribution(actual_values,predicted_values,name:str):

    plt.figure(figsize=(6, 6))
    # Create a joint plot
    sns.set(style="whitegrid")
    joint_plot = sns.jointplot(x=actual_values, y=predicted_values, kind='reg', color='skyblue')

    # Customize the plot
    joint_plot.set_axis_labels('Actual Values', 'Predicted Values', fontsize=12)
    joint_plot.fig.suptitle('Joint Plot of Actual vs Predicted Values', y=1.02, fontsize=14)

    # Define the path where you want to save the plot
    folder_path = "./figures"
    plt.savefig(f"{folder_path}/jointplot {name}.png")
    plt.close()


def predict_plot(model,window_size,df_dict,scaler,name:str):

    df_train = df_dict['df_train']
    df_validation = df_dict['df_val']
    df_test = df_dict['df_test']
    xtest = df_dict['xtest']
    xval = df_dict['xval']

    folder_path = "./figures"

    test_predict = model.predict(xtest)
    val_predict = model.predict(xval)

    # count features, minus one to rest the date column
    num_features = df_train.shape[1]-1

    y_train_unscaled = np.array(df_train['close'].iloc[window_size:])#get_real_scale(np.array(ytrain).reshape(-1, 1),num_features)
    y_test_unscaled = np.array(df_test['close'])
    y_val_unscaled = np.array(df_validation['close'])

    prediction_test = get_real_scale(test_predict,num_features,scaler)
    prediction_val = get_real_scale(val_predict,num_features,scaler)

    mape = calculate_mape(np.array(df_test['close']), np.array(prediction_test))
    rmse = np.sqrt(mean_squared_error(np.array(y_test_unscaled),np.array(prediction_test)))

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"LSTM MAPE is {mape:.2f}%")

    # Plot actual and predicted values for the test set
    parameters = {
    'title': "LSTM Forecasting Test Set",
    'time': df_test['date']
    }

    fig, ax = plt.subplots(figsize=(10,4))

    # Plot results
    ax = plot_series(ax,df_test['close'],parameters,"Real values")
    ax = plot_series(ax,prediction_test,parameters,"LSTM prediction")
    
    plt.savefig(f"{folder_path}/LSTM prediction {name}.png")
    plt.close()

    # Plot the whole dataset to compare the test set at the end of the period
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(df_train['date'], df_train['close'],label="Train")
    ax.plot(df_validation['date'], y_val_unscaled,label="Validation")
    ax.plot(df_test['date'], y_test_unscaled,label="Test")
    ax.plot(df_validation['date'], prediction_val,color='purple')
    ax.plot(df_test['date'], prediction_test,label="Prediction",color='purple')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("LSTM Forecasting")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder_path}/LSTM prediction overview {name}.png")
    plt.close()

    calculate_residuals(y_test_unscaled,prediction_test)
    plot_distribution(y_test_unscaled,prediction_test,name)

    

def forecasting_models(dataset,train_set,test_set):

    # Suppress all DeprecationWarnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    naive_forecaster(train_set,test_set)

    split_val = pd.to_datetime('2023-03-01')
    split_test = pd.to_datetime('2023-04-01')
    window_size = 6

    # Model 1 variable
    new_data = dataset.copy()

    #prepare data
    new_data,split_val_value,split_test_value,data_dict = prepare_data(new_data,split_val,split_test)
    new_data = new_data[['close']].copy()

    xtrain,ytrain,xtest,ytest,xval,yval,scaler = preprocess_data(np.array(new_data),split_val_value,split_test_value,enable_pca=False,window= window_size)

    # Run the model
    print("RUNNNG MODEL 1 - UNIVARIATE")
    history,model = fit_model(xtrain,ytrain,xval,yval)

    data_dict['xtest'] = xtest
    data_dict['xval'] = xval

    # predict with model and plot
    predict_plot(model,window_size,data_dict,scaler,"univariate")

    # Run model 2
    print("RUNNING MODEL 2 - MULTIVARIATE")
    # only used for technical indicators
    new_data = dataset.dropna().reset_index(drop=True)

    #prepare data
    new_data,split_val_value,split_test_value,data_dict = prepare_data(new_data,split_val,split_test)

    #Run with pca
    xtrain,ytrain,xtest,ytest,xval,yval,scaler = preprocess_data(new_data,split_val_value,split_test_value,enable_pca=True,window= window_size)

    # Run the model
    history,model = fit_model(xtrain,ytrain,xval,yval)

    data_dict['xtest'] = xtest
    data_dict['xval'] = xval

    # predict with model and plot
    predict_plot(model,window_size,data_dict,scaler,"multivariate")

