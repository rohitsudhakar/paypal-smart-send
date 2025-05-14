# 01_fetch_and_prepare_data.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from fredapi import Fred
from arch import arch_model
from sklearn.preprocessing import StandardScaler # Keep StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np # Keep numpy
import traceback

# --- Configuration ---
TARGET_TICKER = "INR=X"
INDEX_TICKER = "DX-Y.NYB"
TICKERS = [TARGET_TICKER, INDEX_TICKER]

PERIOD = "2y" # Keep 2y period
INTERVAL = "1h"
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
PREDICT_AHEAD = 1
SEQUENCE_LENGTH = 24 # Keep Seq Len 24

OUTPUT_DIR = "data_processed_lstm_garch_ret" # Directory for returns data

FRED_API_KEY = '50e56aeb469f9f7d16625f81b03cb632'
if FRED_API_KEY == 'YOUR_FRED_API_KEY_HERE': print("WARNING: FRED_API_KEY not set.")
FRED_SERIES = {'DFF': 'Fed Funds', 'DTB3': 'TBill 3M', 'VIXCLS': 'VIX'}

# --- MODIFICATION START: Revert TA Definitions ---
# Remove Stochastic and ADX, keeping only the original set + ATR
TA_INDICATORS_DEF = [
    {"kind": "sma", "length": 10}, {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50},
    {"kind": "rsi", "length": 14}, {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    {"kind": "bbands", "length": 20, "std": 2},
    {"kind": "atr", "length": 14}, # Keep ATR as it was in the original list
]
# --- MODIFICATION END ---
TA_STRATEGY = ta.Strategy(name="TA Strategy", ta=TA_INDICATORS_DEF)
DXY_LAG_FEATURES = [1, 2, 3, 5, 8]

# --- Helper Functions ---
# (Keep calculate_max_lookback, fetch_yf_data, fetch_fred_data as before)
def calculate_max_lookback(indicator_definitions):
    max_lookback = 0
    for indicator in indicator_definitions:
        length = indicator.get('length', 0); period = indicator.get('period', 0); slow = indicator.get('slow', 0); k = indicator.get('k', 0); d = indicator.get('d', 0)
        current_max = max(length, period, slow, k, d) # k, d will be 0 now
        if indicator.get('kind') == 'macd': current_max = max(current_max, indicator.get('fast', 0), indicator.get('signal', 0))
        if current_max > max_lookback: max_lookback = current_max
    print(f"Calculated max lookback from TA_STRATEGY: {max_lookback}")
    return max(max_lookback, 50)

def fetch_yf_data(tickers, period, interval):
    print(f"Fetching {interval} yfinance data for {tickers} over period '{period}'...")
    try:
        data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=False)
        if data.empty: raise ValueError(f"No {interval} data fetched for period {period}.")
        required_ohlc = ['Open', 'High', 'Low', 'Close'] # Still fetch OHLC for ATR
        ohlc_data = pd.DataFrame()
        close_data = pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            if TARGET_TICKER in data.columns.get_level_values(1):
                ohlc_cols = [(col, TARGET_TICKER) for col in required_ohlc if (col, TARGET_TICKER) in data.columns]
                if len(ohlc_cols) == len(required_ohlc):
                     ohlc_data = data[ohlc_cols]; ohlc_data.columns = [col[0] for col in ohlc_cols]
            close_data = data['Close'].copy(); close_data.columns = [f'Close_{ticker}' for ticker in close_data.columns]
        else:
            if all(col in data.columns for col in required_ohlc): ohlc_data = data[required_ohlc].copy()
            close_cols = [col for col in data.columns if 'Close' in col or col == 'Close']
            if not close_cols:
                 if all(t in data.columns for t in tickers): close_cols = tickers; close_data = data[close_cols].copy(); close_data.rename(columns={t: f'Close_{t}' for t in tickers}, inplace=True)
                 else: raise ValueError("Could not find 'Close' columns.")
            else: close_data = data[close_cols].copy()
            if len(tickers) == 1 and 'Close' in close_data.columns and len(close_data.columns)==1: close_data.columns = [f'Close_{tickers[0]}']
        yf_data_to_process = close_data
        if yf_data_to_process.index.tz is None: yf_data_to_process.index = yf_data_to_process.index.tz_localize('UTC')
        else: yf_data_to_process.index = yf_data_to_process.index.tz_convert('UTC')
        if not ohlc_data.empty:
             if ohlc_data.index.tz is None: ohlc_data.index = ohlc_data.index.tz_localize('UTC')
             else: ohlc_data.index = ohlc_data.index.tz_convert('UTC')
             ohlc_data = ohlc_data.reindex(yf_data_to_process.index).ffill().bfill() # Reindex needed here too
        required_cols = [f'Close_{t}' for t in tickers]
        if not all(col in yf_data_to_process.columns for col in required_cols): raise ValueError(f"Required Close columns missing.")
        if yf_data_to_process[required_cols].isnull().values.any():
            print("Warning: yfinance Close NaNs detected. Applying ffill/bfill..."); yf_data_to_process.ffill(inplace=True); yf_data_to_process.bfill(inplace=True);
        if yf_data_to_process[required_cols].isnull().values.any(): raise ValueError("Unfillable NaNs in yfinance Close data.")
        if not ohlc_data.empty and ohlc_data.isnull().values.any():
            print("Warning: yfinance OHLC NaNs detected. Applying ffill/bfill..."); ohlc_data.ffill(inplace=True); ohlc_data.bfill(inplace=True);
        return yf_data_to_process, ohlc_data
    except ValueError as ve: print(f"Error fetching yfinance data: {ve}"); traceback.print_exc(); return None, None
    except Exception as e: print(f"An unexpected error occurred fetching yfinance data: {e}"); traceback.print_exc(); return None, None

def fetch_fred_data(series_dict, start_date, end_date, api_key):
    print(f"Fetching FRED data for series: {list(series_dict.keys())}...")
    if not api_key or api_key == 'YOUR_FRED_API_KEY_HERE': print("FRED API Key not provided. Skipping."); return None
    try:
        fred = Fred(api_key=api_key)
        df_fred = pd.DataFrame()
        for series_id, description in series_dict.items():
            try:
                 s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                 if not s.empty and not s.isnull().all(): df_fred[series_id] = s
                 else: print(f"  Warning: Series {series_id} empty or all NaN.")
            except Exception as series_e: print(f"  Warning: Could not fetch series {series_id}. Error: {series_e}")
        if df_fred.empty: print("Warning: No FRED data fetched successfully."); return None
        df_fred.index = pd.to_datetime(df_fred.index);
        if df_fred.index.tz is None: df_fred.index = df_fred.index.tz_localize('UTC')
        else: df_fred.index = df_fred.index.tz_convert('UTC')
        return df_fred
    except Exception as e: print(f"Error fetching FRED data: {e}"); traceback.print_exc(); return None

# --- MODIFICATION START: Simplify prepare_features_target ---
def prepare_features_target(yf_close_data, yf_ohlc_data, fred_data):
    target_col_name = f'Close_{TARGET_TICKER}'; index_col_name = f'Close_{INDEX_TICKER}'
    if target_col_name not in yf_close_data.columns or index_col_name not in yf_close_data.columns:
        print(f"Error: Required columns '{target_col_name}' or '{index_col_name}' not found in yfinance close data.")
        return None, None, None
    required_ohlc = ['Open', 'High', 'Low', 'Close'] # Still needed for ATR
    use_ohlc_for_ta = False
    if yf_ohlc_data is not None and not yf_ohlc_data.empty and all(col in yf_ohlc_data.columns for col in required_ohlc):
        print("OHLC data found, using it for TA calculations (ATR).")
        use_ohlc_for_ta = True
        # Reindex required even if just for ATR
        yf_ohlc_data = yf_ohlc_data.reindex(yf_close_data.index).ffill().bfill()
    else:
        print("Warning: OHLC data not available or incomplete. Using only Close price for TA calculations (may affect ATR).")
        # If OHLC is missing, create a compatible df for TA strategy call
        yf_ohlc_data = yf_close_data[[target_col_name]].copy()
        yf_ohlc_data.rename(columns={target_col_name: 'Close'}, inplace=True)
        # Add dummy columns if OHLC based indicators are still in strategy (though they shouldn't be now)
        for col in ['Open', 'High', 'Low']:
             if col not in yf_ohlc_data.columns: yf_ohlc_data[col] = yf_ohlc_data['Close']


    print("Merging yfinance Close prices and FRED data...")
    if fred_data is not None and not fred_data.empty:
        if fred_data.index.tz is None: fred_data.index = fred_data.index.tz_localize('UTC')
        elif fred_data.index.tz != yf_close_data.index.tz: fred_data.index = fred_data.index.tz_convert(yf_close_data.index.tz)
        original_yf_index_name = yf_close_data.index.name; original_fred_index_name = fred_data.index.name
        yf_close_data.index.name = 'timestamp'; fred_data.index.name = 'timestamp'
        df_merged = yf_close_data.join(fred_data, how='left')
        yf_close_data.index.name = original_yf_index_name; fred_data.index.name = original_fred_index_name
        df_merged.index.name = original_yf_index_name
        df_merged[fred_data.columns] = df_merged[fred_data.columns].ffill().bfill()
        if df_merged[fred_data.columns].isnull().values.any(): df_merged[fred_data.columns].fillna(0, inplace=True)
    else:
        print("No FRED data to merge."); df_merged = yf_close_data.copy()

    print("Calculating technical indicators (Simplified Set)...")
    df_for_ta = yf_ohlc_data # Use OHLC if available (for ATR), else the prepared Close df
    df_for_ta.ta.cores = 1
    try:
        # Calculate TA using the reverted (simpler) strategy
        df_for_ta.ta.strategy(TA_STRATEGY)
    except Exception as e: print(f"*** Error during ta.strategy call: {e}"); traceback.print_exc(); print("Columns available for TA:", df_for_ta.columns); print("TA Strategy definitions:", TA_STRATEGY.ta); raise

    df_features = pd.DataFrame(index=df_merged.index)
    # Add the calculated (simpler set) TA features
    ta_cols_to_add = [col for col in df_for_ta.columns if col not in required_ohlc]
    print(f"Adding TA features: {ta_cols_to_add}")
    df_features[ta_cols_to_add] = df_for_ta[ta_cols_to_add]

    print("Calculating GARCH volatility forecast feature...")
    returns = df_merged[target_col_name].pct_change().dropna() * 100
    garch_forecast_vol = pd.Series(index=df_merged.index, name='GARCH_Forecast_Vol', dtype=float)
    if not returns.empty and len(returns) > 5:
        garch = arch_model(returns, p=1, q=1, vol='Garch', dist='Normal', mean='Constant')
        try:
            garch_results = garch.fit(disp='off', show_warning=False)
            cond_vol = garch_results.conditional_volatility / 100
            garch_forecast_vol_shifted = cond_vol.shift(1); garch_forecast_vol_shifted.name = 'GARCH_Forecast_Vol'
            df_features['GARCH_Forecast_Vol'] = garch_forecast_vol_shifted
        except Exception as e: print(f"Warning: GARCH fitting failed: {e}. Filling GARCH feature with 0."); df_features['GARCH_Forecast_Vol'] = 0.0
    else: print("Warning: Not enough data points for GARCH calculation. Filling GARCH feature with 0."); df_features['GARCH_Forecast_Vol'] = 0.0
    df_features['GARCH_Forecast_Vol'].fillna(0.0, inplace=True)

    print("Calculating DXY lag features...")
    dxy_pct_change = df_merged[index_col_name].pct_change()
    for lag in DXY_LAG_FEATURES: df_features[f'{index_col_name}_lag_{lag}'] = dxy_pct_change.shift(lag)

    print("Adding FRED features...")
    fred_cols_to_add = [col for col in FRED_SERIES.keys() if col in df_merged.columns]
    if fred_cols_to_add: df_features[fred_cols_to_add] = df_merged[fred_cols_to_add]

    # --- Remove Time Feature Calculation ---
    # print("Adding time features (hour, dayofweek)...")
    # df_features['hour'] = df_features.index.hour; df_features['dayofweek'] = df_features.index.dayofweek
    # df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24.0); df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24.0)
    # df_features['day_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7.0); df_features['day_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7.0)
    print("Skipping time feature calculation.")
    # --- End Remove Time Feature Calculation ---


    print("Adding Target Ticker Close and calculating Target Return...")
    df_features['Target_Ticker_Close'] = df_merged[target_col_name] # Keep based on previous results
    df_features['Target_Return'] = df_merged[target_col_name].pct_change().shift(-PREDICT_AHEAD)

    # --- Define Final Feature Columns ---
    # Automatically select columns present, excluding target and intermediate ones
    feature_columns = [
        col for col in df_features.columns if col not in [
            'Target_Return', 'hour', 'dayofweek', # Exclude target and any raw time cols if accidentally created
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos' # Exclude sin/cos time features explicitly
        ]
    ]
    # Verify the expected base TA cols are present (optional check)
    # expected_base_ta = ['SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'ATRr_14']
    # missing_base = [col for col in expected_base_ta if col not in feature_columns]
    # if missing_base: print(f"Warning: Missing expected base TA columns: {missing_base}")

    print(f"Total features defined ({len(feature_columns)}): {', '.join(sorted(feature_columns))}") # Print sorted list

    initial_rows = df_features.shape[0]
    # Ensure dropna subset contains only columns actually present in df_features
    cols_for_dropna = [col for col in feature_columns + ['Target_Return'] if col in df_features.columns]
    df_features.dropna(subset=cols_for_dropna, inplace=True)
    rows_dropped = initial_rows - df_features.shape[0]
    print(f"Removed {rows_dropped} rows due to NaNs in features or target.")
    if df_features.empty: print("Error: DataFrame empty after NaNs removal."); return None, None, None

    # Ensure X only uses columns that survived the process
    final_feature_columns = [col for col in feature_columns if col in df_features.columns]
    X = df_features[final_feature_columns].copy()
    y = df_features['Target_Return'].copy()
    print("Features (X) and target (y - returns) prepared."); print(f"Final X shape: {X.shape}, Final y shape: {y.shape}")
    global feature_columns_global; feature_columns_global = list(X.columns) # Save the final actual columns
    return X, y, feature_columns_global # Return the final columns used
# --- MODIFICATION END ---


# (Keep create_sequences, split_and_scale_data, save_sequences_etc, plot_data_splits as before)
def create_sequences(X_df, y_series, seq_length):
    X_seq, y_seq = [], []; X_vals = X_df.values; y_vals = y_series.values
    if len(X_vals) <= seq_length: print(f"Error: Data length ({len(X_vals)}) <= sequence length ({seq_length})."); return np.array([]), np.array([])
    for i in range(len(X_vals) - seq_length): X_seq.append(X_vals[i:(i + seq_length)]); y_seq.append(y_vals[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def split_and_scale_data(X, y, feature_columns):
    print("Splitting data chronologically...");
    X_train_val_df, X_test_df, y_train_val_series, y_test_series = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    relative_val_size = VALIDATION_SIZE / (1.0 - TEST_SIZE); X_train_df, X_val_df, y_train_series, y_val_series = train_test_split(X_train_val_df, y_train_val_series, test_size=relative_val_size, shuffle=False)
    print(f"Train shape (pre-seq): {X_train_df.shape}, Val shape: {X_val_df.shape}, Test shape: {X_test_df.shape}"); print("Scaling features using StandardScaler...")
    # Ensure feature_columns passed here matches the columns in X_train_df etc.
    scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_df); X_val_scaled = scaler.transform(X_val_df); X_test_scaled = scaler.transform(X_test_df)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=feature_columns); X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val_df.index, columns=feature_columns); X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_df.index, columns=feature_columns)
    return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train_series, y_val_series, y_test_series, scaler

def save_sequences_etc(output_dir, feature_columns, scaler, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, y_test_series, sequence_length, yf_raw_data):
    os.makedirs(output_dir, exist_ok=True); print(f"Saving processed sequences and artifacts to '{output_dir}'...")
    np.save(os.path.join(output_dir, 'X_train_seq.npy'), X_train_seq); np.save(os.path.join(output_dir, 'y_train_seq_ret.npy'), y_train_seq)
    np.save(os.path.join(output_dir, 'X_val_seq.npy'), X_val_seq); np.save(os.path.join(output_dir, 'y_val_seq_ret.npy'), y_val_seq)
    np.save(os.path.join(output_dir, 'X_test_seq.npy'), X_test_seq); np.save(os.path.join(output_dir, 'y_test_seq_ret.npy'), y_test_seq)
    scaler_filename = os.path.join(output_dir, 'feature_scaler_lstm_garch_ret.joblib'); joblib.dump(scaler, scaler_filename); print(f"Scaler saved to {scaler_filename}")
    feature_list_path = os.path.join(output_dir, 'feature_list.txt'); print(f"Saving {len(feature_columns)} features to {feature_list_path}")
    # Ensure feature_columns being saved is the final correct list
    with open(feature_list_path, 'w') as f: [f.write(f"{feature}\n") for feature in feature_columns]
    np.save(os.path.join(output_dir, 'sequence_length.npy'), np.array([sequence_length])); print(f"Sequence length {sequence_length} saved.")
    if not y_test_series.empty and len(y_test_series) > sequence_length:
        y_test_actual_index = y_test_series.index[sequence_length:]
        pd.Series(y_test_actual_index).to_frame(name='TestIndex').to_parquet(os.path.join(output_dir, 'y_test_index.parquet')); print(f"Test set index saved ({len(y_test_actual_index)} rows).")
        target_col_name_save = f'Close_{TARGET_TICKER}'
        if target_col_name_save in yf_raw_data.columns:
             base_price_indices = y_test_series.index[sequence_length-1 : len(y_test_series)-1]; valid_indices = base_price_indices[base_price_indices.isin(yf_raw_data.index)]
             if len(valid_indices) == len(y_test_seq):
                 y_test_base_prices = yf_raw_data.loc[valid_indices, target_col_name_save]; y_test_base_prices.to_frame(name='BasePrice').to_parquet(os.path.join(output_dir, 'y_test_base_prices.parquet')); print(f"Test set base prices saved ({len(y_test_base_prices)} rows).")
             else: print(f"Warning: Base price saving error. Indices needed: {len(base_price_indices)}, Valid indices found: {len(valid_indices)}, Predictions expected: {len(y_test_seq)}.")
        else: print(f"Warning: Cannot save test base prices. Target column '{target_col_name_save}' not found in raw (Close) data.")
    else: print(f"Warning: y_test_series empty/short vs sequence length ({sequence_length}). Cannot save test index/base prices.")

def plot_data_splits(target_data, X_train_idx, X_val_idx, output_dir):
    if not isinstance(target_data.index, pd.DatetimeIndex): print("Warning: Plotting index is not DatetimeIndex.")
    plt.figure(figsize=(14, 7));
    if 'Close' in target_data.columns: plt.plot(target_data.index, target_data['Close'], label=f'{TARGET_TICKER} Close Price ({INTERVAL})')
    else: print("Warning: 'Close' column not found for plotting.")
    if not X_train_idx.empty: plt.axvline(X_train_idx[-1], color='orange', linestyle='--', label='Train/Val Split')
    if not X_val_idx.empty: plt.axvline(X_val_idx[-1], color='red', linestyle='--', label='Val/Test Split')
    plt.title(f'{TARGET_TICKER} Price ({INTERVAL}) with Splits'); plt.xlabel('Date'); plt.ylabel('Price'); plt.legend(); plt.grid(True)
    plot_filename = os.path.join(output_dir, f'{TARGET_TICKER}_historical_price_splits.png'); plt.savefig(plot_filename); print(f"Saved plot to {plot_filename}"); plt.close()

# --- Global variable for feature columns ---
feature_columns_global = []

# --- Main Execution Guard ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1. Fetch yfinance Data
    yf_close_data, yf_ohlc_data = fetch_yf_data(TICKERS, PERIOD, INTERVAL)
    if yf_close_data is None: exit()
    if not isinstance(yf_close_data.index, pd.DatetimeIndex): print("Error: yfinance data index is not DatetimeIndex."); exit()
    start_date_dt = yf_close_data.index.min().strftime('%Y-%m-%d'); end_date_dt = yf_close_data.index.max().strftime('%Y-%m-%d')
    # 2. Fetch FRED Data
    fred_raw_data = fetch_fred_data(FRED_SERIES, start_date_dt, end_date_dt, FRED_API_KEY)
    # 3. Prepare Features (Simplified Set)
    X_df, y_series, features = prepare_features_target(yf_close_data, yf_ohlc_data, fred_raw_data)
    if X_df is None or y_series is None or features is None: print("Exiting due to error in prepare_features_target."); exit()
    # 4. Split and Scale Data
    X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, y_train_series, y_val_series, y_test_series, scaler = split_and_scale_data(X_df, y_series, features)
    # 5. Create Sequences
    print(f"Creating sequences with length {SEQUENCE_LENGTH}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled_df, y_train_series, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled_df, y_val_series, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled_df, y_test_series, SEQUENCE_LENGTH)
    if X_train_seq.size == 0 or X_val_seq.size == 0 or X_test_seq.size == 0: print("Error: Sequence arrays empty after creation."); exit()
    print(f"Sequence shapes: Train X:{X_train_seq.shape}, Val X:{X_val_seq.shape}, Test X:{X_test_seq.shape}")
    print(f"Sequence shapes: Train y:{y_train_seq.shape}, Val y:{y_val_seq.shape}, Test y:{y_test_seq.shape}")
    # 6. Save Processed Sequences and Artifacts
    save_sequences_etc(OUTPUT_DIR, features, scaler, X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, y_test_series, SEQUENCE_LENGTH, yf_close_data)
    # 7. Plot Original Target Data with Splits
    print("Plotting original price data with splits...")
    target_col_plot = f'Close_{TARGET_TICKER}'
    if target_col_plot in yf_close_data.columns and not X_df.empty:
         plot_data = yf_close_data[[target_col_plot]].loc[X_df.index]; plot_data.rename(columns={target_col_plot: 'Close'}, inplace=True)
         plot_data_splits(plot_data, X_train_scaled_df.index, X_val_scaled_df.index, OUTPUT_DIR)
    else: print("Could not plot original data splits.")

    print("\nData fetching and preparation for LSTM (predicting returns) complete.")
    print(f"Processed data saved in '{OUTPUT_DIR}' with simplified feature set.")