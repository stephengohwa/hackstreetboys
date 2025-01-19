# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:34:43 2025

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:06:44 2025

@author: ASUS
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from difflib import get_close_matches
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from transformers import pipeline
import mplfinance as mpf

# Suppress all FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", ValueWarning)

# Initial settings
initial_capital = 10000.00
margin = 5
fees = 0.08
begin_trading = False
trade_counter = 0  # Counter to track the number of trades

BASE_URL = "https://api.coincap.io/v2"

# Function to fetch a list of available cryptocurrencies and map abbreviations to full IDs
def get_available_cryptos():
    url = f"{BASE_URL}/assets"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {crypto['symbol'].lower(): crypto['id'] for crypto in data['data']}
    else:
        print(f"Error fetching cryptocurrency list: {response.status_code}, {response.text}")
        return {}

def get_closest_crypto(user_input, available_cryptos):
    """
    Suggest the closest matching cryptocurrency based on user input.
    """
    matches = get_close_matches(user_input, available_cryptos.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_market_data(crypto="bitcoin"):
    url = f"{BASE_URL}/assets/{crypto}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching market data: {response.status_code}, {response.text}")
        return None

def fetch_hourly_trends(asset="bitcoin", hours=24):
    """
    Fetch hourly price trends for the last specified 24 hours.
    """
    url = f"{BASE_URL}/assets/{asset}/history"
    params = {
        "interval": "h1",
        "limit": 24  # Fixed to always fetch the last 24 hours
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            df = pd.DataFrame(data['data'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['priceUsd'] = df['priceUsd'].astype(float)
            df.set_index('time', inplace=True)
            df.index = pd.date_range(start=df.index[0], periods=len(df), freq='h')  # Set hourly frequency
            return df
        else:
            print("No data found for the selected cryptocurrency in the given interval.")
            return None
    else:
        print(f"Error fetching hourly trends: {response.status_code}, {response.text}")
        return None

def calculate_moving_averages(data, window=5):
    """
    Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA).
    """
    sma = data['priceUsd'].rolling(window=window).mean()
    ema = data['priceUsd'].ewm(span=window, adjust=False).mean()
    return sma, ema

def calculate_volatility(data):
    """
    Calculate price volatility as the standard deviation of returns.
    """
    returns = data['priceUsd'].pct_change().dropna()
    volatility = np.std(returns) * np.sqrt(len(data))
    return volatility

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = data['priceUsd'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_support_resistance(data):
    """
    Identify support and resistance levels based on local minima and maxima.
    """
    support = data['priceUsd'].min()
    resistance = data['priceUsd'].max()
    return support, resistance

def add_external_features(data):
    """
    Add external features to enhance prediction (e.g., trading volume).
    """
    data['log_volume'] = np.log(data['volume'] + 1)  # Log-transformed volume
    return data

def analyze_sentiment(texts):
    """
    Perform sentiment analysis on a list of texts.
    """
    sentiment_model = pipeline("sentiment-analysis")
    results = sentiment_model(texts)
    return results

def optimize_portfolio(assets):
    """
    Optimize portfolio allocation based on Sharpe ratio.
    """
    returns = [asset['priceUsd'].pct_change().mean() for asset in assets]
    std_devs = [np.std(asset['priceUsd'].pct_change()) for asset in assets]
    sharpe_ratios = [r / s for r, s in zip(returns, std_devs)]
    weights = sharpe_ratios / np.sum(sharpe_ratios)
    return weights

def plot_hourly_trends(df, predicted_value=None, support=None, resistance=None):
    """
    Plot hourly trends using matplotlib, including predicted value as a red line if provided.
    Plot support and resistance levels as horizontal lines.
    """
    if not df.empty:
        ax = df['priceUsd'].plot(kind='line', title="Hourly Price Trends")
        plt.xlabel("Time (24hr Format)")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: df.index[int(x) % len(df.index)].strftime('%H:%M') if int(x) < len(df.index) else ''
            )
        )

        # Plot the predicted value as a red line
        if predicted_value is not None:
            plt.axhline(y=predicted_value, color='red', linestyle='--', label=f"Predicted: ${round(predicted_value, 2)}")

        # Plot support and resistance levels
        if support is not None:
            plt.axhline(y=support, color='green', linestyle='--', label=f"Support: ${round(support, 2)}")
        if resistance is not None:
            plt.axhline(y=resistance, color='orange', linestyle='--', label=f"Resistance: ${round(resistance, 2)}")

        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No data available to plot.")
                

def predict_next_hour_price(data):
    if len(data) < 10:
        print("Insufficient data for prediction. Returning None.")
        return None

    # Step 1: ARIMA prediction
    data.index.freq = 'h'  # Ensure the frequency is explicitly set
    model = ARIMA(data['priceUsd'], order=(5, 1, 0))
    arima_model = model.fit()
    arima_prediction = arima_model.forecast(steps=1).iloc[0]  # Corrected access

    # Step 2: Machine Learning Prediction (Random Forest)
    X = np.arange(len(data)).reshape(-1, 1)  # Feature: Time index
    y = data['priceUsd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    ml_prediction = rf_model.predict([[len(data)]])[0]  # Predict the next point

    # Step 3: Ensemble Prediction
    weights = [0.6, 0.4]  # Tunable weights for ARIMA and ML
    final_prediction = weights[0] * arima_prediction + weights[1] * ml_prediction

    # Confidence bounds
    std_dev = np.std(data['priceUsd'])
    upper_bound = final_prediction + 1.96 * std_dev
    lower_bound = final_prediction - 1.96 * std_dev

    return {
        "prediction": final_prediction,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
    }

def plot_candlestick_chart(data):
    """
    Plot a candlestick chart for the price data.
    """
    try:
        # Ensure the DataFrame has the required columns for OHLC
        ohlc_data = data[['priceUsd']].copy()
        ohlc_data['open'] = ohlc_data['priceUsd'].shift(1)
        ohlc_data['high'] = ohlc_data['priceUsd'].rolling(window=2).max()
        ohlc_data['low'] = ohlc_data['priceUsd'].rolling(window=2).min()
        ohlc_data['close'] = ohlc_data['priceUsd']
        ohlc_data = ohlc_data[['open', 'high', 'low', 'close']].dropna()

        # Plot the candlestick chart with warning suppressed
        mpf.plot(
            ohlc_data,
            type='candle',
            title='Candlestick Chart',
            style='charles',
            ylabel='Price (USD)',
            volume=False,
            warn_too_much_data=1000  # Suppress warning for larger datasets
        )
    except Exception as e:
        print(f"Error in plotting candlestick chart: {e}")

def startbot():
    global begin_trading
    starting = input("Would you like to begin the trading bot? (y/n): ").lower()
    if starting in ['y', 'yes']:
        begin_trading = True
        print("Trading bot started.")
    else:
        print("Trading bot not started. Please try again.")

# Main execution
available_cryptos = get_available_cryptos()
if not available_cryptos:
    print("Unable to fetch cryptocurrency list. Exiting...")
    exit()

while not begin_trading:
    name = input("Please insert your name :) ")
    print(f"Hello, {name}! Welcome to the trading bot!")
    startbot()

while begin_trading:
    trade_counter += 1  # Increment trade counter
    selected_crypto = input("Enter the cryptocurrency name or abbreviation (e.g., 'Bitcoin' or 'BTC'): ").lower()
    closest_match = get_closest_crypto(selected_crypto, available_cryptos)
    if not closest_match:
        print("Invalid cryptocurrency. Please try again.")
        continue

    crypto_id = available_cryptos[closest_match]
    print(f"Using cryptocurrency: {closest_match.capitalize()}")

    market_data = get_market_data(crypto_id)
    if market_data:
        price_usd = float(market_data['data']['priceUsd'])
        change_percent_24hr = float(market_data['data']['changePercent24Hr'])
        print(f"Crypto: {market_data['data']['id']}")
        print(f"Price (USD): ${round(price_usd, 2)}")
        print(f"24h Change: {round(change_percent_24hr, 2)}%")

    trends = fetch_hourly_trends(crypto_id)  # Fixed to always analyze the last 24 hours
    if trends is not None and not trends.empty:
        print("Hourly Price Trends:")
        print(trends[['priceUsd']])

        # Display high, low, average prices, and advanced analytics
        highest_price = trends['priceUsd'].max()
        lowest_price = trends['priceUsd'].min()
        average_price = trends['priceUsd'].mean()
        sma, ema = calculate_moving_averages(trends)
        volatility = calculate_volatility(trends)
        rsi = calculate_rsi(trends)
        support, resistance = calculate_support_resistance(trends)

        print(f"Highest Price: ${round(highest_price, 2)}")
        print(f"Lowest Price: ${round(lowest_price, 2)}")
        print(f"Average Price: ${round(average_price, 2)}")
        print(f"SMA (5 hours): {sma[-1]:.2f}")
        print(f"EMA (5 hours): {ema[-1]:.2f}")
        print(f"Volatility: {volatility:.4f}")
        print(f"RSI (14 hours): {rsi[-1]:.2f}")
        print(f"Support Level: ${support:.2f}")
        print(f"Resistance Level: ${resistance:.2f}")

        # Predict next hour price
        prediction = predict_next_hour_price(trends)
        if prediction:
            print(f"Predicted Price (Next Hour): ${round(prediction['prediction'], 2)}")
            print(f"Confidence Interval: ${round(prediction['upper_bound'], 2)} - ${round(prediction['lower_bound'], 2)}")

            # Bad bot logic
            if prediction['prediction'] < price_usd:
                print(f"Prediction is lower than the current price. Suggestion: BUY NOW at ${round(price_usd, 2)}!")
                print(f"-"*100)
            else:
                print(f"Prediction is higher than the current price. Suggestion: SELL NOW at ${round(price_usd, 2)}!")
                print(f"-"*100)

            # Plot trends with predicted value and support/resistance levels
            plot_hourly_trends(trends, predicted_value=prediction['prediction'], support=support, resistance=resistance)
    else:
        print("Hourly trends data unavailable. Try another cryptocurrency or time range.")
        
        if trends is not None and not trends.empty:
            print("Hourly Price Trends:")
            print(trends[['priceUsd']])

    # Display the candlestick chart
    plot_candlestick_chart(trends)

    # Plot trends with predicted value and support/resistance levels
    plot_hourly_trends(trends, predicted_value=prediction['prediction'], support=support, resistance=resistance)

    # Prompt user after 5 trades
    if trade_counter % 5 == 0:
        cont = input("Would you like to continue trading? (y/n): ").lower()
        if cont not in ['y', 'yes']:
            print("Stopping trading bot.")
            begin_trading = False

print("Thank you for using the trading bot!")


#byhackstreetboys
