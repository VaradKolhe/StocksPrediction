import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objs as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Prediction App", layout="wide")

# --- COMPANY MAPPING ---
COMPANY_MAP = {
    "Apple Inc.": "AAPL",
    "Google (Alphabet)": "GOOGL",
    "Microsoft Corp.": "MSFT",
    "Tesla Inc.": "TSLA",
    "Amazon.com": "AMZN",
    "NVIDIA Corp.": "NVDA",
    "Netflix": "NFLX",
    "Meta (Facebook)": "META",
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "Samsung Electronics": "005930.KS",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Toyota Motor": "7203.T",
    "Sony Group": "6758.T"
}

# --- CURRENCY SYMBOL MAPPING ---
def get_currency_symbol(currency_code):
    symbols = {
        "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹",
        "JPY": "¥", "KRW": "₩", "CNY": "¥", "AUD": "A$",
        "CAD": "C$", "SGD": "S$", "CHF": "Fr"
    }
    return symbols.get(currency_code, currency_code + " ") # Default to code if symbol not found

# --- HELPER FUNCTIONS ---

def get_stock_info(ticker):
    """Fetches currency and name metadata."""
    try:
        info = yf.Ticker(ticker).info
        currency = info.get('currency', 'USD')
        symbol = get_currency_symbol(currency)
        return symbol
    except:
        return "$" # Default fallback

def load_data(ticker):
    """Fetches live data from Yahoo Finance."""
    data = yf.download(ticker, period="5y")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    return data

def add_technical_features(df):
    """Applies EXACT Feature Engineering logic."""
    df = df.copy()
    
    # 1. Standardize names
    df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    # 2. Basic Calculations
    df["daily_return"] = df["close"].pct_change()
    df["price_range"] = df["high"] - df["low"]
    df["open_close_diff"] = df["open"] - df["close"]

    # 3. Lags
    df["lag_1"] = df["close"].shift(1)
    df["lag_5"] = df["close"].shift(5)
    df["lag_10"] = df["close"].shift(10)

    # 4. Moving Averages
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # 5. Volatility
    df["volatility_10"] = df["daily_return"].rolling(10).std()

    # 6. EMA & MACD
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]

    # 7. RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 8. Normalized Close
    df["close_norm"] = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())

    df.dropna(inplace=True)
    return df

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home (Prophet Forecast)", "Real-Time Linear Regression"])

# ==========================================
# PAGE 1: PROPHET FORECAST
# ==========================================
if page == "Home (Prophet Forecast)":
    st.title("📈 Long-Term Trend Forecasting")
    st.markdown("Select a company to predict future trends over months or years.")
    
    selected_name = st.selectbox("Select Company", list(COMPANY_MAP.keys()))
    ticker = COMPANY_MAP[selected_name] 

    years = st.slider("Years of prediction:", 1, 4)
    period = years * 365

    if st.button("Forecast"):
        with st.spinner(f"Fetching data for {selected_name}..."):
            try:
                data = load_data(ticker)
                if data.empty:
                    st.error("No data found.")
                else:
                    df_train = data[['Date', 'Close']]
                    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                    
                    df_train['ds'] = pd.to_datetime(df_train['ds'])
                    if df_train['ds'].dt.tz is not None:
                        df_train['ds'] = df_train['ds'].dt.tz_localize(None)

                    m = Prophet()
                    m.fit(df_train)
                    future = m.make_future_dataframe(periods=period)
                    forecast = m.predict(future)

                    st.subheader(f"Forecast for {selected_name}")
                    fig1 = plot_plotly(m, forecast)
                    st.plotly_chart(fig1)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# ==========================================
# PAGE 2: LINEAR REGRESSION (Multi-Company)
# ==========================================
elif page == "Real-Time Linear Regression":
    st.title("🤖 Real-Time Next-Day Predictor")
    st.markdown("Predicts the **next trading day's price** using live technical indicators.")

    selected_names = st.multiselect(
        "Select Companies to Analyze", 
        list(COMPANY_MAP.keys()), 
        default=["Apple Inc.", "Samsung Electronics"]
    )

    if st.button("Train & Predict Selected"):
        if not selected_names:
            st.warning("Please select at least one company.")
        
        summary_data = [] 

        for name in selected_names:
            ticker = COMPANY_MAP[name]
            
            # 1. Get Currency Symbol (e.g., $, ₹, ₩)
            currency_symbol = get_stock_info(ticker)
            
            with st.expander(f"📊 Analysis for {name} ({ticker})", expanded=True):
                with st.spinner(f"Processing {name}..."):
                    try:
                        # 2. Fetch Data
                        raw_data = load_data(ticker)
                        
                        # Fix Date Index for Plotting
                        if "Date" in raw_data.columns:
                            raw_data["Date"] = pd.to_datetime(raw_data["Date"])
                            raw_data.set_index("Date", inplace=True)

                        if raw_data.empty:
                            st.error(f"No data found for {name}.")
                        else:
                            # 3. Feature Engineering
                            df = add_technical_features(raw_data)
                            
                            FEATURES = [
                                "daily_return", "price_range", "open_close_diff",
                                "lag_1", "lag_5", "lag_10",
                                "ma_5", "ma_20",
                                "volatility_10",
                                "ema_12", "ema_26", "macd",
                                "rsi", "close_norm"
                            ]
                            
                            if len(df) < 50:
                                st.error(f"Not enough data for {name}.")
                            else:
                                # 4. Train/Test Logic
                                df['target_close_tomorrow'] = df['close'].shift(-1)
                                train_df = df.dropna()
                                
                                X = train_df[FEATURES]
                                y = train_df['target_close_tomorrow']

                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_scaled, y, test_size=0.2, shuffle=False
                                )

                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                
                                # Metrics
                                y_pred_test = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred_test)
                                mae = mean_absolute_error(y_test, y_pred_test)

                                # 5. Predict Next Day
                                latest_features = df.iloc[[-1]][FEATURES]
                                latest_features_scaled = scaler.transform(latest_features)
                                predicted_price = model.predict(latest_features_scaled)[0]
                                current_price = df.iloc[-1]['close']
                                
                                direction = "UP" if predicted_price > current_price else "DOWN"
                                color = "green" if direction == "UP" else "red"
                                pct_change = ((predicted_price - current_price) / current_price) * 100

                                # Save for Summary Table (Formatted with correct symbol)
                                summary_data.append({
                                    "Company": name,
                                    "Current Price": f"{currency_symbol}{current_price:,.2f}",
                                    "Predicted Price": f"{currency_symbol}{predicted_price:,.2f}",
                                    "Direction": direction,
                                    "Change %": f"{pct_change:.2f}%"
                                })

                                # --- DISPLAY METRICS ---
                                m1, m2, m3 = st.columns(3)
                                m1.metric("R² Score", f"{r2:.2%}")
                                m2.metric("MAE (Error)", f"{currency_symbol}{mae:.2f}")
                                m3.metric("Current Price", f"{currency_symbol}{current_price:,.2f}")

                                st.markdown(f"""
                                #### Forecast: <span style='color:{color}'>{currency_symbol}{predicted_price:,.2f}</span>
                                """, unsafe_allow_html=True)

                                # --- PLOT ---
                                fig = go.Figure()
                                subset_indices = df.index[len(X_train):len(X_train)+len(X_test)][-60:]
                                subset_y_test = y_test[-60:]
                                subset_y_pred = y_pred_test[-60:]

                                fig.add_trace(go.Scatter(x=subset_indices, y=subset_y_test, name='Actual', line=dict(color='blue')))
                                fig.add_trace(go.Scatter(x=subset_indices, y=subset_y_pred, name='Predicted', line=dict(color='orange', dash='dot')))
                                
                                next_day = df.index[-1] + pd.Timedelta(days=1)
                                fig.add_trace(go.Scatter(
                                    x=[df.index[-1], next_day], 
                                    y=[current_price, predicted_price],
                                    name='Next Day Forecast',
                                    mode='lines+markers',
                                    line=dict(color=color, width=3)
                                ))
                                fig.update_layout(height=400, margin=dict(t=30, b=20), hovermode="x unified")
                                st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error processing {name}: {e}")
        
        if summary_data:
            st.markdown("### 📋 Summary of Predictions")
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True)