import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def calculate_cagr(start_price, end_price, start_date, end_date):
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    if start_price <= 0 or end_price <= 0:
        return 0.0 

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    delta = end_date - start_date
    num_years = delta.days / 365.25 

    if num_years <= 0:
        return 0.0

    try:
        return ((end_price / start_price)**(1/num_years)) - 1
    except ZeroDivisionError:
        return 0.0

# ==========================================
# 2. DATA LOADING (MATCHING YOUR SAVING ORDER)
# ==========================================

@st.cache_data
def load_data():
    """
    Loads the pre-trained model data from forecast_data.pkl.
    EXPECTED ORDER: (stocks_fe, lr, xgb, sarima)
    """
    file_path = 'forecast_data.pkl'
    
    if not os.path.exists(file_path):
        return None, None, None, None
        
    try:
        with open(file_path, 'rb') as f:
            # Data is loaded in the order you saved it:
            # 1. stocks_fe (Historical)
            # 2. lr_future_predictions (Linear Regression)
            # 3. xgb_future_predictions (XGBoost)
            # 4. sarima_future_predictions (SARIMA)
            data = pickle.load(f)
            
            stocks_fe = data[0]
            lr_future = data[1]
            xgb_future = data[2]
            sarima_future = data[3]
            
        return stocks_fe, lr_future, xgb_future, sarima_future
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None, None, None, None

# Load data
stocks_fe, lr_future_predictions, xgb_future_predictions, sarima_future_predictions = load_data()

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title('Stock Price Forecast Dashboard')

# Check if data loaded
if stocks_fe is None:
    st.error("File 'forecast_data.pkl' not found.")
    st.info("Run the pickle save code in your notebook, then move the resulting .pkl file here.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# 1. Company Selection
available_companies = list(stocks_fe.keys())
selected_company = st.sidebar.selectbox("Select Company", available_companies)

# 2. Prediction Model Selection
model_options = ['Linear Regression', 'XGBoost', 'SARIMA']
selected_model_name = st.sidebar.selectbox("Prediction Model", model_options)

# Map selection to data
model_mapping = {
    'Linear Regression': lr_future_predictions,
    'XGBoost': xgb_future_predictions,
    'SARIMA': sarima_future_predictions
}
selected_model_dict = model_mapping[selected_model_name]

# 3. Dynamic Year Slider
if selected_company in stocks_fe:
    last_hist_date = stocks_fe[selected_company].index[-1]
    min_forecast_year = last_hist_date.year + 1
else:
    min_forecast_year = datetime.now().year

forecast_year = st.sidebar.slider(
    "Select Forecast Year", 
    min_value=int(min_forecast_year), 
    max_value=2030, 
    value=min(2025, 2030) # Default to 2025 or max if history is recent
)

# --- Main Content Area ---
has_history = selected_company in stocks_fe
has_forecast = (selected_model_dict is not None) and (selected_company in selected_model_dict)

if not has_history:
    st.error(f"No historical data available for {selected_company}.")
elif not has_forecast:
    st.warning(f"No forecast data available for {selected_company} using {selected_model_name}.")
    # Show history anyway
    st.line_chart(stocks_fe[selected_company]['close'])
else:
    df_history = stocks_fe[selected_company]
    df_forecast_full = selected_model_dict[selected_company]
    
    # Filter forecast date up to selected year
    end_date_str = f"{forecast_year}-12-31"
    
    # 1. Force the index to be Datetime objects
    df_forecast_full.index = pd.to_datetime(df_forecast_full.index)

    # 2. Ensure your end_date is also a datetime object
    end_date = pd.to_datetime(end_date_str)
    
    # 3. Now the comparison will work
    df_forecast = df_forecast_full[df_forecast_full.index <= end_date]

    if df_forecast.empty:
        st.warning(f"No forecast data found up to {forecast_year}.")
    else:
        # --- Plotting ---
        st.subheader(f"{selected_company}: History vs {selected_model_name}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot History
        ax.plot(df_history.index, df_history['close'], label='Historical Close', color='#1f77b4', linewidth=2)
        
        # Plot Prediction
        ax.plot(df_forecast.index, df_forecast['predicted_close'], label='Predicted Close', color='#ff7f0e', linestyle='--', linewidth=2)
        
        ax.set_title(f"Stock Price Forecast (Target: {forecast_year})", fontsize=14)
        ax.set_ylabel("Price ($)")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # --- Metrics ---
        last_hist_p = df_history['close'].iloc[-1]
        last_pred_p = df_forecast['predicted_close'].iloc[-1]
        
        cagr = calculate_cagr(last_hist_p, last_pred_p, df_history.index[-1], df_forecast.index[-1])
        growth_pct = (last_pred_p - last_hist_p) / last_hist_p

        st.markdown("### Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${last_hist_p:,.2f}", help="Last available historical closing price")
            
        with col2:
            st.metric(
                f"Price in {forecast_year}", 
                f"${last_pred_p:,.2f}", 
                delta=f"{growth_pct:.1%}",
                help=f"Predicted price at the end of {forecast_year}"
            )
            
        with col3:
            st.metric("Implied CAGR", f"{cagr:.2%}", help="Compound Annual Growth Rate")