# 04_predict_lstm.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from fredapi import Fred
from arch import arch_model
import joblib
import os
from datetime import datetime, timedelta
import traceback
from curl_cffi import requests
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings # To handle the FutureWarning

# --- Configuration ---
TARGET_TICKER = "INR=X"
INDEX_TICKER = "DX-Y.NYB"
TICKERS = [TARGET_TICKER, INDEX_TICKER]

INTERVAL = "1h"
# Increase lookback slightly to ensure enough data for TA indicators like ADX/Stoch
LOOKBACK_PERIOD = "60d" # Increased from 30d, needed for TA lookbacks + GARCH

PROCESSED_DATA_DIR = "data_processed_lstm_garch_ret" # Data with GARCH features
MODEL_SAVE_DIR = "models"
SCALER_FILENAME = os.path.join(PROCESSED_DATA_DIR, 'feature_scaler_lstm_garch_ret.joblib')
FEATURE_LIST_FILENAME = os.path.join(PROCESSED_DATA_DIR, 'feature_list.txt')
MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, "lstm_usdinr_model_fred_garch_ret.h5") # Use the latest trained model
SEQ_LEN_FILENAME = os.path.join(PROCESSED_DATA_DIR, 'sequence_length.npy')

FRED_API_KEY = '50e56aeb469f9f7d16625f81b03cb632'
FRED_SERIES = {'DFF': 'Fed Funds', 'DTB3': 'TBill 3M', 'VIXCLS': 'VIX'}

# --- MODIFICATION START: Update TA Definitions ---
# Ensure TA definitions match those used in 01_fetch_and_prepare_data.py
TA_INDICATORS_DEF = [
    {"kind": "sma", "length": 10}, {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50},
    {"kind": "rsi", "length": 14}, {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    {"kind": "bbands", "length": 20, "std": 2}, {"kind": "atr", "length": 14},
    # Add Stochastic and ADX
    {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
    {"kind": "adx", "length": 14},
]
TA_STRATEGY = ta.Strategy(name="Predict TA", ta=TA_INDICATORS_DEF)
# --- MODIFICATION END ---
DXY_LAG_FEATURES = [1, 2, 3, 5, 8]

# --- Helper Functions ---
# (Keep calculate_max_lookback, fetch_fred_data_recent)
def calculate_max_lookback(indicator_definitions):
    max_lookback = 0
    for indicator in indicator_definitions:
        length = indicator.get('length', 0); period = indicator.get('period', 0); slow = indicator.get('slow', 0); k = indicator.get('k', 0); d = indicator.get('d', 0)
        current_max = max(length, period, slow, k, d)
        if indicator.get('kind') == 'macd': current_max = max(current_max, indicator.get('fast', 0), indicator.get('signal', 0))
        if current_max > max_lookback: max_lookback = current_max
    print(f"Calculated max lookback for prediction TA: {max_lookback}")
    return max(max_lookback, 50) # Use safe minimum

def fetch_fred_data_recent(series_dict, api_key, lookback_days=60):
    print(f"Fetching recent FRED data (last ~{lookback_days} days)...")
    if not api_key or api_key == 'YOUR_FRED_API_KEY_HERE': return None
    try:
        fred = Fred(api_key=api_key)
        end_date = datetime.utcnow().date(); start_date = end_date - timedelta(days=lookback_days)
        df_fred = pd.DataFrame()
        for series_id, description in series_dict.items():
            try:
                 s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                 if not s.empty and not s.isnull().all(): df_fred[series_id] = s
                 else: print(f"  Warning: Series {series_id} empty/NaN for recent period.")
            except Exception as series_e: print(f"  Warning: Could not fetch series {series_id}. Error: {series_e}")
        if df_fred.empty: print("Warning: No recent FRED data fetched successfully."); return None
        df_fred.index = pd.to_datetime(df_fred.index).tz_localize('UTC')
        print("Recent FRED data fetched.")
        return df_fred
    except Exception as e: print(f"Error fetching recent FRED data: {e}"); traceback.print_exc(); return None

# --- MODIFICATION START: Update fetch_yf_data for prediction ---
def fetch_yf_data_predict(tickers, period, interval):
    """Fetches yfinance data, ensuring OHLC is available if possible."""
    print(f"Fetching {interval} yfinance data for {tickers} over period '{period}'...")
    try:
        unsafe_session = requests.Session(impersonate="chrome")
        unsafe_session.verify = False
        data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=False, session=unsafe_session)
        if data.empty: raise ValueError(f"No {interval} data fetched for period {period}.")

        required_ohlc = ['Open', 'High', 'Low', 'Close']
        ohlc_data = pd.DataFrame()
        close_data = pd.DataFrame()

        # Handle MultiIndex columns (common for multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Extract OHLC for the target ticker if possible
            if TARGET_TICKER in data.columns.get_level_values(1):
                ohlc_cols = [(col, TARGET_TICKER) for col in required_ohlc if (col, TARGET_TICKER) in data.columns]
                if len(ohlc_cols) == len(required_ohlc):
                    ohlc_data = data[ohlc_cols].copy()
                    ohlc_data.columns = [col[0] for col in ohlc_cols] # Simplify column names

            # Always extract Close prices for all tickers
            close_data = data['Close'].copy()
            close_data.columns = [f'Close_{ticker}' for ticker in close_data.columns]

        else: # Handle single index columns (e.g., single ticker fetch)
            if all(col in data.columns for col in required_ohlc):
                ohlc_data = data[required_ohlc].copy() # Assume OHLC are present

            # Extract Close prices
            close_cols = [col for col in data.columns if 'Close' in col or col == 'Close']
            if not close_cols:
                 if all(t in tickers for t in data.columns): close_cols = tickers; close_data = data[close_cols].copy(); close_data.rename(columns={t: f'Close_{t}' for t in tickers}, inplace=True)
                 else: raise ValueError("Could not find 'Close' columns.")
            else: close_data = data[close_cols].copy()
            # Ensure column naming consistency if only one ticker
            if len(tickers) == 1 and 'Close' in close_data.columns and len(close_data.columns)==1: close_data.columns = [f'Close_{tickers[0]}']

        # --- Process and return ---
        yf_data_to_process = close_data # Use Close prices for merging, lags, GARCH, target

        # Ensure timezone consistency
        if yf_data_to_process.index.tz is None: yf_data_to_process.index = yf_data_to_process.index.tz_localize('UTC')
        else: yf_data_to_process.index = yf_data_to_process.index.tz_convert('UTC')
        if not ohlc_data.empty:
             if ohlc_data.index.tz is None: ohlc_data.index = ohlc_data.index.tz_localize('UTC')
             else: ohlc_data.index = ohlc_data.index.tz_convert('UTC')
             # Reindex OHLC to match Close data index, filling any gaps
             ohlc_data = ohlc_data.reindex(yf_data_to_process.index).ffill().bfill()


        # Check for NaNs in required Close columns
        required_cols = [f'Close_{t}' for t in tickers]
        if not all(col in yf_data_to_process.columns for col in required_cols): raise ValueError(f"Required Close columns missing.")
        if yf_data_to_process[required_cols].isnull().values.any():
            print("Warning: yfinance Close NaNs detected. Applying ffill/bfill...");
            yf_data_to_process.ffill(inplace=True); yf_data_to_process.bfill(inplace=True);
        if yf_data_to_process[required_cols].isnull().values.any(): raise ValueError("Unfillable NaNs in yfinance Close data.")

        # Fill NaNs in OHLC data if it exists
        if not ohlc_data.empty and ohlc_data.isnull().values.any():
            print("Warning: yfinance OHLC NaNs detected. Applying ffill/bfill...");
            ohlc_data.ffill(inplace=True); ohlc_data.bfill(inplace=True);

        print(f"yfinance data fetched successfully. Close shape: {yf_data_to_process.shape}, OHLC shape: {ohlc_data.shape}")
        return yf_data_to_process, ohlc_data

    except ValueError as ve:
        print(f"Error fetching yfinance data: {ve}")
        traceback.print_exc(); return None, None
    except Exception as e:
        print(f"An unexpected error occurred fetching yfinance data: {e}")
        traceback.print_exc(); return None, None
# --- MODIFICATION END ---


# --- Main execution block ---
if __name__ == '__main__':

    # --- 1. Load Model, Scaler, Features, Sequence Length ---
    print("Loading artifacts...")
    try:
        scaler = joblib.load(SCALER_FILENAME)
        with open(FEATURE_LIST_FILENAME, 'r') as f:
            feature_columns = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(feature_columns)} feature names from {FEATURE_LIST_FILENAME}")
        SEQUENCE_LENGTH = int(np.load(SEQ_LEN_FILENAME)[0])
        print(f"Loaded sequence length: {SEQUENCE_LENGTH}")

        # Check if the model is an HDF5 file
        if MODEL_FILENAME.endswith(".h5"):
             # Define custom objects if needed (e.g., custom loss/metrics used during training)
             # Since we used 'mae' string, it might load automatically, but being explicit is safer.
             custom_objects = {'mae': tf.keras.losses.MeanAbsoluteError}
             # Load HDF5 model
             model = load_model(MODEL_FILENAME, custom_objects=custom_objects)
             print(f"Loaded Keras HDF5 model: {MODEL_FILENAME}")
        else:
             # Assume TensorFlow SavedModel format if not HDF5
             model = load_model(MODEL_FILENAME)
             print(f"Loaded Keras SavedModel: {MODEL_FILENAME}")

    except FileNotFoundError as e: print(f"Error loading necessary file: {e}"); exit()
    except Exception as e: print(f"An error occurred during loading: {e}"); traceback.print_exc(); exit()
    print("Artifacts loaded.")

    # --- 2. Fetch Latest yfinance and FRED Data ---
    print(f"\nFetching latest {INTERVAL} data for {TICKERS} (period: {LOOKBACK_PERIOD})...")
    try:
        # --- MODIFICATION START: Use updated fetch function ---
        latest_yf_close_data, latest_yf_ohlc_data = fetch_yf_data_predict(TICKERS, LOOKBACK_PERIOD, INTERVAL)
        if latest_yf_close_data is None: exit() # Exit if essential close data failed
        # --- MODIFICATION END ---

        # Merge FRED data
        latest_fred_data = fetch_fred_data_recent(FRED_SERIES, FRED_API_KEY, lookback_days=90) # Fetch slightly more FRED history

        if latest_fred_data is not None and not latest_fred_data.empty:
            if latest_fred_data.index.tz is None: latest_fred_data.index = latest_fred_data.index.tz_localize('UTC')
            elif latest_fred_data.index.tz != latest_yf_close_data.index.tz: latest_fred_data.index = latest_fred_data.index.tz_convert(latest_yf_close_data.index.tz)
            # Merge using Close data index
            latest_data_merged = latest_yf_close_data.join(latest_fred_data, how='left')
            fred_cols = latest_fred_data.columns
            latest_data_merged[fred_cols] = latest_data_merged[fred_cols].ffill().bfill()
            if latest_data_merged[fred_cols].isnull().values.any(): latest_data_merged[fred_cols].fillna(0, inplace=True)
        else:
            print("Proceeding without FRED data."); latest_data_merged = latest_yf_close_data.copy()
            # Add dummy FRED columns if they are expected features but not fetched
            for fred_s_id in FRED_SERIES.keys():
                 if fred_s_id in feature_columns and fred_s_id not in latest_data_merged.columns: latest_data_merged[fred_s_id] = 0

        # --- Minimum rows check ---
        ta_max_lookback = calculate_max_lookback(TA_INDICATORS_DEF)
        min_garch_window = 50 # GARCH needs fewer points for prediction than initial fit
        min_rows_needed = SEQUENCE_LENGTH + max(ta_max_lookback, max(DXY_LAG_FEATURES), min_garch_window) + 2
        if latest_data_merged.shape[0] < min_rows_needed:
            raise ValueError(f"Insufficient data ({latest_data_merged.shape[0]} rows) fetched for lookback {LOOKBACK_PERIOD}. Need ~{min_rows_needed}. Try increasing LOOKBACK_PERIOD.")

        print(f"Latest data fetched/merged: {latest_data_merged.shape[0]} rows, up to {latest_data_merged.index.max()}")

    except Exception as e: print(f"Error fetching/processing latest data: {e}"); traceback.print_exc(); exit()


    # --- 3. Calculate Features for Latest Data Period (incl. GARCH, new TA, Time) ---
    print("Calculating features for the latest data period...")
    try:
        target_col_name = f'Close_{TARGET_TICKER}'; index_col_name = f'Close_{INDEX_TICKER}'

        # --- MODIFICATION START: Calculate TA using fetched OHLC if available ---
        required_ohlc = ['Open', 'High', 'Low', 'Close']
        use_ohlc_for_ta_pred = False
        df_for_ta_pred = pd.DataFrame()

        if latest_yf_ohlc_data is not None and not latest_yf_ohlc_data.empty and all(col in latest_yf_ohlc_data.columns for col in required_ohlc):
             print("Using fetched OHLC data for prediction TA.")
             use_ohlc_for_ta_pred = True
             # Ensure OHLC data is aligned with the merged data index
             df_for_ta_pred = latest_yf_ohlc_data.reindex(latest_data_merged.index).ffill().bfill()
        else:
             print("Warning: OHLC data not available for prediction. Using Close price only for TA.")
             df_for_ta_pred = latest_data_merged[[target_col_name]].copy()
             df_for_ta_pred.rename(columns={target_col_name: 'Close'}, inplace=True) # Needs 'Close' column

        # Calculate TA indicators
        df_for_ta_pred.ta.cores = 1
        df_for_ta_pred.ta.strategy(TA_STRATEGY)
        # --- MODIFICATION END ---


        # --- Start building features dataframe ---
        df_features_latest = pd.DataFrame(index=latest_data_merged.index)

        # Add TA features calculated above
        ta_cols_pred = [col for col in df_for_ta_pred.columns if col not in required_ohlc]
        print(f"Adding prediction TA features: {ta_cols_pred}")
        df_features_latest[ta_cols_pred] = df_for_ta_pred[ta_cols_pred]

        # --- Fit GARCH and forecast volatility ---
        print("Fitting GARCH and forecasting volatility...")
        returns = latest_data_merged[target_col_name].pct_change().dropna() * 100
        garch_forecast_vol = pd.Series(index=latest_data_merged.index, name='GARCH_Forecast_Vol', dtype=float)
        if len(returns) >= min_garch_window: # Use min_garch_window check
             garch_pred = arch_model(returns, p=1, q=1, vol='Garch', dist='Normal', mean='Constant')
             try:
                 # Fit model up to second to last observation to forecast the last one needed
                 garch_results = garch_pred.fit(last_obs=returns.index[-2], disp='off', show_warning=False)
                 # Forecast one step ahead from the end of the fitted period
                 forecast = garch_results.forecast(horizon=1, reindex=False)
                 predicted_variance = forecast.variance.iloc[0, 0] # Get the numeric variance value
                 # Convert variance to volatility (std dev) and assign to the *next* timestamp
                 garch_forecast_vol.loc[returns.index[-1]] = np.sqrt(predicted_variance) / 100 # Assign forecast to last index
                 # Backfill earlier GARCH values using conditional volatility from fit
                 cond_vol_fit = garch_results.conditional_volatility / 100
                 garch_forecast_vol = garch_forecast_vol.combine_first(cond_vol_fit.shift(1)) # Shifted fit results
                 garch_forecast_vol.name = 'GARCH_Forecast_Vol' # Ensure name is correct
             except Exception as garch_e:
                 print(f"Warning: GARCH fit/forecast failed during prediction: {garch_e}. Filling with 0.")
                 garch_forecast_vol.fillna(0.0, inplace=True) # Fill everything if GARCH fails
        else:
             print(f"Warning: Not enough return data ({len(returns)}) for GARCH. Filling with 0.")

        # Ensure GARCH column exists and fill NaNs (e.g., first value)
        df_features_latest['GARCH_Forecast_Vol'] = garch_forecast_vol
        df_features_latest['GARCH_Forecast_Vol'].fillna(0.0, inplace=True)


        # --- Add other features ---
        # DXY lags
        df_features_latest[f'{index_col_name}_pct_change'] = latest_data_merged[index_col_name].pct_change()
        for lag in DXY_LAG_FEATURES:
            df_features_latest[f'{index_col_name}_lag_{lag}'] = df_features_latest[f'{index_col_name}_pct_change'].shift(lag)
        # Drop the intermediate pct_change column if not needed as feature
        if f'{index_col_name}_pct_change' not in feature_columns:
             df_features_latest = df_features_latest.drop(columns=[f'{index_col_name}_pct_change'])


        # FRED features
        fred_cols_in_features = [col for col in FRED_SERIES.keys() if col in feature_columns]
        if fred_cols_in_features:
            # Make sure FRED cols exist in merged data, add 0 if not
            for col in fred_cols_in_features:
                if col not in latest_data_merged.columns: latest_data_merged[col] = 0
            df_features_latest[fred_cols_in_features] = latest_data_merged[fred_cols_in_features]

        # --- MODIFICATION START: Add Time Features ---
        print("Adding time features for prediction...")
        df_features_latest['hour'] = df_features_latest.index.hour
        df_features_latest['dayofweek'] = df_features_latest.index.dayofweek
        df_features_latest['hour_sin'] = np.sin(2 * np.pi * df_features_latest['hour'] / 24.0)
        df_features_latest['hour_cos'] = np.cos(2 * np.pi * df_features_latest['hour'] / 24.0)
        df_features_latest['day_sin'] = np.sin(2 * np.pi * df_features_latest['dayofweek'] / 7.0)
        df_features_latest['day_cos'] = np.cos(2 * np.pi * df_features_latest['dayofweek'] / 7.0)
        # Drop raw hour/dayofweek if not in feature_columns
        if 'hour' not in feature_columns: df_features_latest = df_features_latest.drop(columns=['hour'])
        if 'dayofweek' not in feature_columns: df_features_latest = df_features_latest.drop(columns=['dayofweek'])
        # --- MODIFICATION END ---

        # Target Ticker Close (if it's a feature)
        if 'Target_Ticker_Close' in feature_columns:
            df_features_latest['Target_Ticker_Close'] = latest_data_merged[target_col_name]

        # --- Final Checks ---
        # Reindex df_features_latest to ensure all expected columns are present before dropna
        df_features_latest = df_features_latest.reindex(columns=feature_columns, fill_value=np.nan)

        # Check for missing columns *after* attempting to add all features
        missing_cols = [col for col in feature_columns if col not in df_features_latest.columns]
        if missing_cols:
            # This shouldn't happen if reindex is done correctly, but check anyway
            raise ValueError(f"Columns missing after reindex: {missing_cols}.")

        print(f"Columns in df_features_latest before dropna: {df_features_latest.columns.tolist()}")

        # Drop rows with NaNs (important after adding lags and TA)
        initial_rows_pred = df_features_latest.shape[0]
        df_features_latest.dropna(inplace=True)
        rows_dropped_pred = initial_rows_pred - df_features_latest.shape[0]
        print(f"Removed {rows_dropped_pred} rows from prediction features due to NaNs.")

        if df_features_latest.empty or len(df_features_latest) < SEQUENCE_LENGTH:
             raise ValueError(f"Not enough valid rows ({len(df_features_latest)}) after calculating features and dropping NaNs for sequence length {SEQUENCE_LENGTH}. Try increasing LOOKBACK_PERIOD.")

        print(f"Final prediction features calculated for {len(df_features_latest)} rows.")
        # Select the exact features required by the model, in the correct order
        last_features_unscaled = df_features_latest[feature_columns].iloc[-SEQUENCE_LENGTH:]

        if last_features_unscaled.isnull().any().any(): raise ValueError("NaNs found in final feature sequence for prediction.")
        if len(last_features_unscaled) != SEQUENCE_LENGTH: raise ValueError(f"Final sequence length error. Expected {SEQUENCE_LENGTH}, got {len(last_features_unscaled)}.")

    except Exception as e: print(f"Error calculating features: {e}"); traceback.print_exc(); exit()


    # --- 4. Scale Latest Features ---
    print("Scaling latest features sequence...")
    try:
        # Ensure columns are in the same order as when the scaler was fitted
        last_sequence_scaled = scaler.transform(last_features_unscaled[feature_columns])
        X_pred_seq = last_sequence_scaled.reshape((1, SEQUENCE_LENGTH, len(feature_columns)))
        print(f"Input sequence shape for prediction: {X_pred_seq.shape}")
    except Exception as e: print(f"Error scaling features: {e}"); traceback.print_exc(); exit()

    # --- 5. Make Prediction (Predicts Return) ---
    print("Making prediction (predicting return)...")
    try:
        # Use the loaded model to predict
        predicted_return = model.predict(X_pred_seq)[0][0]
    except Exception as e: print(f"Error during LSTM prediction: {e}"); traceback.print_exc(); exit()

    # --- 6. Convert Predicted Return to Price ---
    print("Converting predicted return to price level...")
    try:
        # Get the last known price from the *original fetched data* corresponding to the end of the sequence
        last_known_price_timestamp = last_features_unscaled.index[-1]
        last_known_price = latest_yf_close_data.loc[last_known_price_timestamp, f'Close_{TARGET_TICKER}'] # Use original close data
        predicted_price = last_known_price * (1 + predicted_return)
    except KeyError:
         print(f"Error: Timestamp {last_known_price_timestamp} not found in latest_yf_close_data. Cannot get last known price.")
         traceback.print_exc(); predicted_price=np.nan
    except Exception as e:
         print(f"Error converting return to price: {e}"); traceback.print_exc(); predicted_price=np.nan

    # --- 7. Display Results ---
    last_timestamp = last_features_unscaled.index[-1]
    if isinstance(last_timestamp, pd.Timestamp):
        # Predict ahead based on interval (assuming 1 hour here)
        # More robustly, infer frequency from index
        freq = pd.infer_freq(last_features_unscaled.index)
        if freq: prediction_timestamp = last_timestamp + pd.Timedelta(hours=1) # Adjust if interval changes
        else: prediction_timestamp = last_timestamp + pd.Timedelta(hours=1) # Default to 1 hour if freq unknown
        ts_string = prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'); last_ts_string = last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
    else: ts_string = f"Next Step (TS Error)"; last_ts_string = str(last_timestamp) + " (TS Error)"

    print("\n--- Prediction Results ---")
    print(f"Based on data sequence ending at: {last_ts_string}")
    print(f"Last known actual price          : {last_known_price:.4f}")
    print(f"Predicted Return for next step   : {predicted_return:.6f} ({predicted_return*100:.4f}%)")
    print(f"Prediction for approx timestamp  : {ts_string}")
    print(f"Predicted Close Price (Estimate) : {predicted_price:.4f}")
    print("---------------------------------------------------------")