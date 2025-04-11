# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import asyncio
from PIL import Image
from main import (
    get_stock_data, get_news_sentiment, predict_stock_price,
    plot_stock_with_predictions, calculate_prediction_metrics,
    get_company_info
)

# Set minimalist page config
st.set_page_config(layout="wide", page_title="Bull Vision", initial_sidebar_state="expanded")

# App logo + title
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("Bull Vision.png")
    st.image(logo, width=100)
with col2:
    st.markdown("<h1 style='font-size: 2.5rem;'>📈 Bull Vision: Stock Forecast & Insights</h1>", unsafe_allow_html=True)

# Sidebar
page = st.sidebar.selectbox("📂 Choose a page", ["Stock Analysis", "Price Prediction"])
company_name = st.sidebar.text_input("🔍 Enter Stock Ticker (e.g., NVDA, AAPL)")

# Show company info box if ticker is entered
if company_name:
    with st.expander("📄 Company Overview"):
        try:
            info = get_company_info(company_name)
            st.markdown(f"""
                **{info['Name']}**  
                *{info['Sector']} | {info['Industry']}*  
                **CEO:** {info['CEO']}  
                **Country:** {info['Country']}    
                **Website:** [{info['Website']}]({info['Website']})  
                <br>
                {info['Description'][:500]}...
            """, unsafe_allow_html=True)
        except:
            st.warning("Could not retrieve company information.")

# Pages
if page == "Stock Analysis":
    st.markdown("### 🧠 Stock Analysis Dashboard")
    if company_name:
        with st.spinner("⏳ Fetching stock data..."):
            df = get_stock_data(company_name)

        if df is not None and not df.empty:
            st.subheader("📈 OHLC Candlestick")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}",
                          f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%")
            with col2:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            with col3:
                st.metric("52-Week Range", f"${df['Low'].min():.2f} - ${df['High'].max():.2f}")

            # Technical charts
            st.subheader("📉 Technical Indicators")
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df[['Close', 'SMA', 'EMA']])
            with col2:
                st.line_chart(df[['Close', 'VWAP']])

            # Sentiment
            st.subheader("📰 News Sentiment")
            with st.spinner("Analyzing sentiment..."):
                sentiment = asyncio.run(get_news_sentiment(company_name))
            sentiment_color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
            st.markdown(
                f"<h3 style='color: {sentiment_color.get(sentiment, 'black')}'>Overall Sentiment: {sentiment}</h3>",
                unsafe_allow_html=True
            )

            with st.expander("📁 Show Raw Data"):
                st.dataframe(df.tail(10))
        else:
            st.error(f"Could not fetch data for '{company_name}'. Please check the symbol.")
    else:
        st.info("👈 Enter a stock ticker to begin.")

elif page == "Price Prediction":
    st.markdown("### 🔮 Future Price Prediction")
    if company_name:
        col1, col2 = st.columns(2)
        with col1:
            prediction_days = st.slider("📆 Days to Predict", 5, 90, 30)
        with col2:
            confidence_interval = st.checkbox("📉 Show Confidence Interval", value=True)

        with st.spinner("🔄 Generating predictions..."):
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

                st.subheader("📏 Prediction Metrics")
                metrics = calculate_prediction_metrics(df, hist_pred_prices)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("MSE", metrics["MSE"])
                with col2: st.metric("MAE", metrics["MAE"])
                with col3: st.metric("MAPE", f"{metrics['MAPE']}%")

                st.subheader("📋 Prediction Summary")
                current_price = df['Close'].iloc[-1]
                future_price = future_pred_prices[-1][0]
                percent_change = ((future_price - current_price) / current_price) * 100
                st.success(f"{company_name} is predicted to {'increase' if percent_change > 0 else 'decrease'} by {abs(percent_change):.2f}% in {prediction_days} days to ${future_price:.2f}")

                with st.expander("📊 Forecast Details"):
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_pred_prices.flatten()
                    })
                    forecast_df['Date'] = forecast_df['Date'].dt.date
                    forecast_df['Predicted Price'] = forecast_df['Predicted Price'].round(2)
                    st.dataframe(forecast_df)

                if st.button("📤 Export to CSV"):
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{company_name}_prediction.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.error("Stock data unavailable.")
    else:
        st.info("👈 Enter a stock ticker to begin prediction.")
