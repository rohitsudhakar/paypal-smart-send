# generate_mock_data.py
# Run this script directly to create/overwrite data/fx_history.csv
# with more varied simulated data patterns.
# Example: python generate_mock_data.py

import os
import random
import pandas as pd
from datetime import datetime, timedelta, timezone

# --- Configuration ---
NUM_HOURS = 72 # Generate data for the past 72 hours
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fx_history.csv")

# **** UPDATED PARAMETERS FOR MORE DIVERSITY ****
# Define pairs and their characteristics (Base Rate, Trend Factor, Volatility)
# Trend Factor: Controls strength & direction (+ve up, -ve down) relative to base rate over the period.
# Volatility: Controls random noise range.
PAIR_CONFIG = {
    # Strong Up -> Should often trigger "wait"
    "GBPJPY": {"base": 194.50, "trend": 0.30, "vol": 0.40},  # Strong Up, Volatile
    "GBPUSD": {"base": 1.2650, "trend": 0.0030, "vol": 0.0015}, # Strong Up
    # Moderate Up -> Might trigger "wait" or "uncertain"
    "USDINR": {"base": 83.55, "trend": 0.015, "vol": 0.02},  # Moderate Up
    "GBPINR": {"base": 105.70,"trend": 0.01, "vol": 0.06},   # Added slight Up Trend to volatile pair
    # Strong Down -> Should often trigger "send_now"
    "EURINR": {"base": 90.15, "trend": -0.030, "vol": 0.03},  # Stronger Down
    "USDJPY": {"base": 153.50, "trend": -0.30, "vol": 0.25},   # Stronger Down
    # Moderate Down -> Might trigger "send_now" or "uncertain"
    "EURUSD": {"base": 1.0750, "trend": -0.0015, "vol": 0.0008}, # Moderate Down
    "EURJPY": {"base": 165.00, "trend": -0.10, "vol": 0.30},   # Moderate Down, Volatile
    # Flat / Volatile Flat -> Should trigger "stable" or "uncertain"
    "EURGBP": {"base": 0.8500, "trend": 0.0, "vol": 0.0008}, # Flat, moderately volatile
    # Very Stable -> Should trigger "stable"
    "JPYINR": {"base": 0.545, "trend": 0.0002, "vol": 0.0005}, # Very low trend/vol
}

# --- Helper Function ---
def get_precision_for_pair(pair, base_rate=1.0):
    precision = 4;
    if "JPY" in pair: precision = 3
    elif pair in ["EURUSD", "GBPUSD", "EURGBP"]: precision = 5
    elif base_rate < 1 and base_rate != 0: precision = 5
    return precision

def generate_mock_history(pair, base_rate, trend_factor, volatility, num_points=NUM_HOURS):
    """ Generates a list of [Timestamp, Pair, Rate] lists for one pair. """
    history = []; current_rate = base_rate; now = datetime.now(timezone.utc); precision = get_precision_for_pair(pair, base_rate); total_points_generated = num_points + 6
    for i in range(total_points_generated, 0, -1):
        timestamp = now - timedelta(hours=(i - 6)); noise = random.uniform(-volatility, volatility); trend_effect = trend_factor * (total_points_generated - i) / total_points_generated; mean_reversion = (base_rate - current_rate) * 0.05; current_rate += trend_effect + mean_reversion + noise; current_rate = max(current_rate, base_rate * 0.80); current_rate = min(current_rate, base_rate * 1.20); history.append([timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'), pair, round(current_rate, precision)])
    return history[-num_points:]

# --- Main Generation Logic ---
if __name__ == "__main__":
    all_data = []
    print(f"Generating more varied mock data for {NUM_HOURS} hours...")

    for pair, config in PAIR_CONFIG.items():
        print(f"  Generating {pair} (Trend={config['trend']}, Vol={config['vol']})...")
        pair_history = generate_mock_history(
            pair, config["base"], config["trend"], config["vol"]
        )
        all_data.extend(pair_history)

    if not all_data:
        print("No data generated.")
    else:
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=["Timestamp", "Pair", "Rate"])
        df['Timestamp'] = pd.to_datetime(df['Timestamp']) # Ensure datetime objects
        df = df.sort_values(by=["Timestamp", "Pair"]) # Sort

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save to CSV
        try:
            df.to_csv(OUTPUT_FILE, index=False, date_format='%Y-%m-%dT%H:%M:%SZ') # Use ISO format
            print(f"\nSuccessfully generated and saved varied mock data to {OUTPUT_FILE}")
            print(f"Total records generated: {len(df)}")
            print("\nSample data (last 5 rows):")
            print(df.tail())
        except Exception as e:
            print(f"Error saving data to {OUTPUT_FILE}: {e}")