# Bull Vision: Stock Insights & Prediction

## Overview
Bull Vision is a Streamlit-based application that provides advanced stock market analysis and forecasting. It retrieves historical stock data, computes technical indicators, performs news sentiment analysis, and forecasts future stock prices using a custom LSTM neural network model. With interactive visualizations powered by Plotly, users can easily explore stock trends and prediction results.

## Features

### Historical Data Retrieval
- Fetches up to six months of historical data for a given stock ticker via the Yahoo Finance API using the yfinance library.

### Technical Indicators
- Calculates key technical metrics including the Simple Moving Average (SMA), Exponential Moving Average (EMA), and Volume Weighted Average Price (VWAP) utilizing the ta library.

### News Sentiment Analysis
- Gathers news articles related to a stock from NewsAPI and uses TextBlob to compute an overall sentiment polarity, helping you gauge the current market mood.

### Stock Price Prediction
- Prepares historical data and leverages a custom LSTM model (built with TensorFlow/Keras) to forecast future stock prices.
- The model is trained on historical data (if a pre-trained model isn't found) and then stored locally as `lstm_stock_model.h5`.

### Interactive Data Visualization
- Provides interactive candlestick charts, technical indicator overlays, and forecast graphs using Plotly, all integrated within a responsive Streamlit UI.

### Data Export
- Allows users to export detailed prediction results as CSV files for further analysis.

## Prerequisites
- Python 3.6+
- Required Libraries:
  - yfinance
  - pandas
  - numpy
  - aiohttp
  - datetime
  - streamlit
  - plotly
  - python-dotenv
  - textblob
  - beautifulsoup4
  - ta
  - scikit-learn
  - tensorflow

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sophomore-seminar-project.git
   cd sophomore-seminar-project
   ```

2. **Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project's root directory with your NewsAPI key:
   ```
   NEWSAPI_KEY=your_newsapi_key_here
   ```
   The `.env` file is automatically loaded to provide secure configuration for news sentiment analysis.

## Usage

1. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```
   This command opens the application in your default web browser.

2. **Navigate the Application**:

   - **Stock Analysis**:
     - Enter a valid stock ticker (e.g., AAPL or TSLA) using the sidebar.
     - View the interactive candlestick chart, detailed technical indicators, and overall news sentiment analysis.
     - Inspect raw data by expanding the "Show Raw Data" section.

   - **Price Prediction**:
     - Choose the number of days to predict the future stock price.
     - See the model's forecast displayed alongside the historical data.
     - Review prediction accuracy metrics (MSE, MAE, MAPE) and download the detailed forecast as a CSV file.

## Code Breakdown

### `app.py`:
Contains the Streamlit user interface. It allows users to select between "Stock Analysis" and "Price Prediction" pages and displays interactive charts and metrics.

### `main.py`:
Houses the core functionality for data processing and analysis:
- Retrieves historical stock data and calculates technical indicators (SMA, EMA, VWAP).
- Performs asynchronous news sentiment analysis using NewsAPI and TextBlob.
- Prepares data for an LSTM model, trains the model if necessary, and forecasts future stock prices.
- Generates interactive Plotly charts for visualizing the predictions and historical trends.

## Future Improvements

### Enhanced Model Tuning
- Experiment with different LSTM architectures or incorporate additional market data to further improve forecast accuracy.

### User Customizations
- Allow users to select custom technical indicator parameters and set personalized alert thresholds.

### Expanded Data Sources
- Integrate more comprehensive data feeds and additional APIs to enrich the analysis.

## License
[Include your chosen license here, e.g., MIT License.]

## Acknowledgements
- Special thanks to the developers of key libraries including yfinance, Streamlit, Plotly, TensorFlow/Keras, and ta.
- Appreciation to NewsAPI for supplying the news data that powers the sentiment analysis.

## Contact
For any questions, suggestions, or contributions, please open an issue on this repository or contact the repository owner directly.