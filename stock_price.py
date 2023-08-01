import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def create_features_and_labels(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_stock_prices(model, X_test):
    return model.predict(X_test)

def plot_predictions(actual_prices, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices.index, actual_prices, label='Actual Prices')
    plt.plot(actual_prices.index[len(actual_prices) - len(predicted_prices):], predicted_prices, label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction using Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'  # Ticker symbol of the stock you want to predict
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    window_size = 30  # Number of previous days to use as features

    # Step 1: Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Step 2: Create features and labels
    X, y = create_features_and_labels(stock_data, window_size)

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    model = train_linear_regression_model(X_train, y_train)

    # Step 5: Make predictions
    predicted_prices = predict_stock_prices(model, X_test)

    # Step 6: Plot the predictions
    plot_predictions(stock_data, predicted_prices)
