# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import asyncio
from main import (
    get_stock_data, get_news_sentiment, predict_stock_price
    , plot_stock_with_predictions, calculate_prediction_metrics
)

st.set_page_config(layout="wide", page_title="Bull Vision")
st.title("📈 Stock Insights - Get Advanced Stock Analysis & Prediction")

# Sidebar for navigation and inputs
page = st.sidebar.selectbox("Choose a page", ["Stock Analysis", "Price Prediction"])
company_name = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")

if page == "Stock Analysis":
    st.header("📊 Stock Analysis Dashboard")
    if company_name:
        with st.spinner("Fetching stock data..."):
            df = get_stock_data(company_name)
        
        if df is not None and not df.empty:
            st.subheader(f"{company_name} Stock Price")
            # Candlestick chart for OHLC data
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            ))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}",
                          f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%")
            with col2:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            with col3:
                st.metric("52-Week Range", f"${df['Low'].min():.2f} - ${df['High'].max():.2f}")
            
            st.subheader("Technical Indicators")
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df[['Close', 'SMA', 'EMA']])
            with col2:
                st.line_chart(df[['Close', 'VWAP']])
            
            st.subheader("📰 News Sentiment Analysis")
            with st.spinner("Analyzing news sentiment..."):
                sentiment = asyncio.run(get_news_sentiment(company_name))
            sentiment_color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
            st.markdown(
                f"<h3 style='color: {sentiment_color.get(sentiment, 'black')}'>Overall Sentiment: {sentiment}</h3>",
                unsafe_allow_html=True
            )
            
            with st.expander("Show Raw Data"):
                st.dataframe(df.tail(10))

        else:
            st.error(f"Could not fetch data for ticker '{company_name}'. Please check the ticker symbol.")
    else:
        st.info("👈 Enter a stock ticker in the sidebar to begin analysis")

elif page == "Price Prediction":
    st.header("🔮 Stock Price Prediction")
    if company_name:
        col1, col2 = st.columns(2)
        with col1:
            prediction_days = st.slider("Days to predict into future", 5, 90, 30)
        with col2:
            confidence_interval = st.checkbox("Show confidence interval", value=True)
        
        with st.spinner("Generating prediction model..."):
            df = get_stock_data(company_name)
            if df is not None and not df.empty:
                if confidence_interval:
                    (hist_pred_prices, hist_pred_dates, future_pred_prices, 
                     future_dates, lower_bounds, upper_bounds) = predict_stock_price(df, prediction_days, return_confidence=True)
                    fig = plot_stock_with_predictions(df, company_name, hist_pred_prices, hist_pred_dates, future_pred_prices, future_dates, lower_bounds, upper_bounds)
                else:
                    hist_pred_prices, hist_pred_dates, future_pred_prices, future_dates = predict_stock_price(df, prediction_days)
                    fig = plot_stock_with_predictions(df, company_name, hist_pred_prices, hist_pred_dates, future_pred_prices, future_dates)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Prediction Accuracy Metrics")
                metrics = calculate_prediction_metrics(df, hist_pred_prices)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Squared Error", metrics["MSE"])
                with col2:
                    st.metric("Mean Absolute Error", metrics["MAE"])
                with col3:
                    st.metric("Mean Absolute % Error", f"{metrics['MAPE']}%")
                
                st.subheader("Prediction Summary")
                current_price = df['Close'].iloc[-1]
                future_price = future_pred_prices[-1][0]
                percent_change = ((future_price - current_price) / current_price) * 100
                if percent_change > 0:
                    st.success(f"The model predicts {company_name} will increase by {percent_change:.2f}% in {prediction_days} days, reaching ${future_price:.2f}")
                else:
                    st.error(f"The model predicts {company_name} will decrease by {abs(percent_change):.2f}% in {prediction_days} days, reaching ${future_price:.2f}")
                
                with st.expander("View Detailed Price Forecast"):
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_pred_prices.flatten()
                    })
                    forecast_df['Date'] = forecast_df['Date'].dt.date
                    forecast_df['Predicted Price'] = forecast_df['Predicted Price'].round(2)
                    st.dataframe(forecast_df, use_container_width=True)
                
                if st.button("Export Predictions"):
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{company_name}_price_prediction.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.error(f"Could not fetch data for ticker '{company_name}'. Please check the ticker symbol.")
    else:
        st.info("👈 Enter a stock ticker in the sidebar to begin prediction")
