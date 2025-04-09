# analysis.py

import os
import yfinance as yf
import numpy as np
import datetime
import aiohttp
from dotenv import load_dotenv
from textblob import TextBlob
import ta  # Technical Analysis Library
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def get_stock_data(company_name):
    ticker = yf.Ticker(company_name)
    df = ticker.history(period='6mo')
    df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        df['High'], df['Low'], df['Close'], df['Volume']
    ).volume_weighted_average_price()
    return df

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
async def fetch_news(session, url):
    async with session.get(url) as response:
        # Return JSON data from the response
        return await response.json()

async def get_news_sentiment(company_name):
    # Check whether NEWSAPI_KEY is set or not
    if not NEWSAPI_KEY:
        raise ValueError("NEWSAPI_KEY is not set in your environment variables.")
    
    # Construct the URL for NewsAPI using "everything" endpoint.
    # The query searches for articles related to the company and its stock.
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={company_name}+stock&"
        "language=en&"
        "pageSize=10&"
        f"apiKey={NEWSAPI_KEY}"
    )
    
    # Create an asynchronous session and fetch the news JSON data.
    async with aiohttp.ClientSession() as session:
        data = await fetch_news(session, url)
    
    # Extract articles from the response. NewsAPI returns a JSON object
    # with 'articles' as one of the keys.
    articles = data.get('articles', [])
    
    # Compute sentiment polarity using the 'description' if available,
    # otherwise fallback to 'title'.
    sentiment_scores = [
        TextBlob(article.get('description') or article.get('title', '')).sentiment.polarity
        for article in articles
    ]
    
    # Calculate the average sentiment polarity.
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    
    # Return a sentiment category based on the average polarity.
    return "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"

def prepare_lstm_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(30, len(df_scaled)):
        X.append(df_scaled[i-30:i, 0])
        y.append(df_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    return model

def predict_stock_price(df, days_to_predict=30, return_confidence = False):
    X, y, scaler = prepare_lstm_data(df)
    model_path = "lstm_stock_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = train_lstm(X, y)
        model.save(model_path)
    
    # Historical predictions
    historical_predictions = model.predict(X)
    historical_pred_prices = scaler.inverse_transform(historical_predictions)
    historical_pred_dates = df.index[-len(historical_predictions):]
    
    # Future predictions
    last_sequence = X[-1].reshape(1, X.shape[1], 1)
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    last_date = df.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days_to_predict)]
    
    for _ in range(days_to_predict):
        pred = model.predict(current_sequence)[0]
        future_predictions.append(pred)
        current_sequence = np.append(current_sequence[:, 1:, :], [[pred]], axis=1)
    
    future_pred_array = np.array(future_predictions).reshape(-1, 1)
    future_pred_prices = scaler.inverse_transform(future_pred_array)
    
    if return_confidence:
        # Compute the error on historical predictions
        actual_hist = df['Close'].values[-len(historical_pred_prices):]
        residuals = actual_hist - historical_pred_prices.flatten()
        std_error = np.std(residuals)
        # 95% confidence interval: ±1.96 * standard error
        lower_bounds = future_pred_prices - 1.96 * std_error
        upper_bounds = future_pred_prices + 1.96 * std_error
        return historical_pred_prices, historical_pred_dates, future_pred_prices, future_dates, lower_bounds, upper_bounds

    return historical_pred_prices, historical_pred_dates, future_pred_prices, future_dates


def plot_stock_with_predictions(df, company_name, hist_pred_prices, hist_pred_dates, future_pred_prices, future_dates, lower_bounds=None, upper_bounds=None):
    fig = go.Figure()

    # Actual historical prices
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        mode='lines', 
        name='Actual Price',
        line=dict(color='blue')
    ))
    
    # Historical predictions
    fig.add_trace(go.Scatter(
        x=hist_pred_dates, 
        y=hist_pred_prices.flatten(), 
        mode='lines', 
        name='Model Fit',
        line=dict(color='green', dash='dot')
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_pred_prices.flatten(), 
        mode='lines', 
        name='Price Forecast',
        line=dict(color='red')
    ))
    
    # Plot confidence interval if provided
    if lower_bounds is not None and upper_bounds is not None:
        ci_x = list(future_dates) + list(future_dates[::-1])
        ci_y = list(upper_bounds.flatten()) + list(lower_bounds.flatten()[::-1])
        fig.add_trace(go.Scatter(
            x=ci_x,
            y=ci_y,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',  # light red fill for CI
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="95% Confidence Interval"
        ))
    
    # Add range selector and layout updates
    fig.update_layout(
        title=f"{company_name} Stock Price Prediction",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title="Price (USD)"),
        height=600
    )
    
    return fig


def calculate_prediction_metrics(df, predicted_prices):
    actual_prices = df['Close'].values[-len(predicted_prices):]
    mse = np.mean((actual_prices - predicted_prices.flatten())**2)
    mae = np.mean(np.abs(actual_prices - predicted_prices.flatten()))
    mape = np.mean(np.abs((actual_prices - predicted_prices.flatten()) / actual_prices)) * 100
    return {
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2)
    }
