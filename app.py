# app.py
import uuid
import random
import math
from flask import Flask, render_template, request, jsonify, url_for
from datetime import datetime, timedelta, timezone
import numpy as np

app = Flask(__name__)

# --- Enhanced Mock Data & Simulation ---

SUPPORTED_CURRENCIES = ["USD", "EUR", "INR", "JPY", "GBP"]

MOCK_FX_HISTORY = {}
# More sophisticated mock history generation (e.g., adding volatility clustering concept)
def generate_mock_history(base_rate, trend_factor, base_volatility, num_points=72): # Simulate 72 hours
    history = []
    current_rate = base_rate
    current_volatility = base_volatility
    now = datetime.now(timezone.utc)
    for i in range(num_points + 6, 0, -1):
        timestamp = now - timedelta(hours=(i - 6))

        # Simulate changing volatility (simple GARCH-like effect simulation)
        if random.random() < 0.1: # Chance to change volatility regime
            current_volatility = base_volatility * random.uniform(0.5, 1.5)
        current_volatility = max(0.0001, current_volatility * 0.95 + base_volatility * 0.05) # Mean revert volatility

        noise = random.gauss(0, current_volatility) # Use Gaussian noise

        # Simulate trend with some momentum concept
        trend_effect = trend_factor * (num_points - i) / num_points
        if i < num_points - 2: # Add simple momentum from previous steps
             momentum = (history[-1][1] - history[-2][1]) * 0.1
             trend_effect += momentum

        mean_reversion = (base_rate - current_rate) * 0.03 # Slightly weaker mean reversion

        current_rate += trend_effect + mean_reversion + noise
        current_rate = max(current_rate, base_rate * 0.80) # Wider bounds
        current_rate = min(current_rate, base_rate * 1.20)

        precision = 6 if base_rate < 10 else (3 if "JPY" in pair else 5) # More dynamic precision storage
        history.append((timestamp, round(current_rate, precision)))

    return history[-(num_points + 1):]

# Generate history using the enhanced function
pairs_to_generate = set()
for c1 in SUPPORTED_CURRENCIES:
    for c2 in SUPPORTED_CURRENCIES:
        if c1 != c2: pairs_to_generate.add(f"{c1}{c2}")

base_rates_params = { # Adjusted volatilities slightly
    "USDINR": {"rate": 83.55, "trend": 0.005, "vol": 0.002}, # INR vol lower relative
    "EURINR": {"rate": 90.15, "trend": -0.008, "vol": 0.003},
    "GBPINR": {"rate": 105.70, "trend": 0.0, "vol": 0.0035},
    "JPYINR": {"rate": 0.545, "trend": 0.001, "vol": 0.0005},
    "EURUSD": {"rate": 1.0750, "trend": -0.0005, "vol": 0.0006},
    "GBPUSD": {"rate": 1.2650, "trend": 0.0010, "vol": 0.0007},
    "USDJPY": {"rate": 153.50, "trend": 0.1, "vol": 0.2},
    "EURGBP": {"rate": 0.8500, "trend": -0.0002, "vol": 0.0004},
    "EURJPY": {"rate": 165.00, "trend": 0.05, "vol": 0.15},
    "GBPJPY": {"rate": 194.50, "trend": 0.08, "vol": 0.25}
}
generated_pairs = set()
# (Generation logic remains similar, just uses the new function)
for pair in pairs_to_generate:
    if pair in generated_pairs: continue
    inverse_pair = f"{pair[3:]}{pair[:3]}"
    params = base_rates_params.get(pair)
    inverse_params = base_rates_params.get(inverse_pair)
    if params:
        MOCK_FX_HISTORY[pair] = generate_mock_history(params["rate"], params["trend"], params["vol"])
        generated_pairs.add(pair); generated_pairs.add(inverse_pair)
    elif inverse_params:
        MOCK_FX_HISTORY[inverse_pair] = generate_mock_history(inverse_params["rate"], inverse_params["trend"], inverse_params["vol"])
        generated_pairs.add(pair); generated_pairs.add(inverse_pair)
    else:
        print(f"Warning: No direct params for {pair} or {inverse_pair}. Using fallback.")
        fallback_rate = 1.0 # Add better fallback logic if needed
        MOCK_FX_HISTORY[pair] = generate_mock_history(fallback_rate, 0, 0.0001)
        generated_pairs.add(pair); generated_pairs.add(inverse_pair)


# --- More Detailed Mock Fundamental/Economic Data ---
FUNDAMENTAL_DATA = {
    "USD": {"name": "US", "interest_rate": 5.25, "inflation": 3.1, "gdp_growth": 1.8, "unemployment": 3.9, "outlook": "neutral", "vol_index": 15},
    "EUR": {"name": "Eurozone", "interest_rate": 4.50, "inflation": 2.8, "gdp_growth": 0.5, "unemployment": 6.5, "outlook": "weak", "vol_index": 14},
    "INR": {"name": "India", "interest_rate": 6.50, "inflation": 5.1, "gdp_growth": 7.0, "unemployment": 7.8, "outlook": "strong", "vol_index": 18},
    "JPY": {"name": "Japan", "interest_rate": -0.10, "inflation": 2.5, "gdp_growth": 0.8, "unemployment": 2.6, "outlook": "neutral", "vol_index": 16},
    "GBP": {"name": "UK", "interest_rate": 5.25, "inflation": 4.0, "gdp_growth": 0.2, "unemployment": 4.2, "outlook": "mixed", "vol_index": 15},
}
# Mock Technical Indicators (Simple Example)
TECHNICAL_INDICATORS = { pair: {"rsi": random.uniform(30, 70), "macd_signal": random.choice([-1, 0, 1])} for pair in MOCK_FX_HISTORY.keys()}

# Mock News with Impact Scores
MOCK_NEWS = [
    {"headline": "Fed hints at slower pace of rate cuts amid sticky inflation.", "sentiment": 0.3, "impact": 0.7, "affects": ["USD"]},
    {"headline": "India's service sector activity hits 6-month high.", "sentiment": 0.8, "impact": 0.8, "affects": ["INR"]},
    {"headline": "German IFO Business Climate index lower than expected.", "sentiment": -0.6, "impact": 0.6, "affects": ["EUR"]},
    {"headline": "Bank of England holds rates, inflation concerns remain.", "sentiment": 0.1, "impact": 0.5, "affects": ["GBP"]},
    {"headline": "Global risk aversion rises, boosting safe-haven currencies.", "sentiment": 0.0, "impact": 0.4, "affects": ["USD", "JPY"]}, # Neutral sentiment, market effect
    {"headline": "Japan's export growth slows.", "sentiment": -0.4, "impact": 0.5, "affects": ["JPY"]},
]

def get_current_rate_and_details(pair):
    """Gets the latest rate, ensuring appropriate precision for calculations and display."""
    ccy1, ccy2 = pair[:3], pair[3:]
    rate_value = None
    data_pair = None # The key used to retrieve history (direct or inverse)

    if pair in MOCK_FX_HISTORY and MOCK_FX_HISTORY[pair]:
        rate_value = MOCK_FX_HISTORY[pair][-1][1]
        data_pair = pair
    else:
        inverse_pair = f"{ccy2}{ccy1}"
        if inverse_pair in MOCK_FX_HISTORY and MOCK_FX_HISTORY[inverse_pair]:
            last_inverse_rate = MOCK_FX_HISTORY[inverse_pair][-1][1]
            if last_inverse_rate != 0:
                rate_value = 1.0 / last_inverse_rate
                data_pair = inverse_pair # Note that inverse data was used

    if rate_value is None:
        print(f"Warning: No direct or inverse data found for {pair}")
        return None, None, None

    # Determine display precision based on the QUOTE currency (ccy2)
    display_precision = 4
    if ccy2 == "JPY": display_precision = 2
    if pair in ["EURUSD", "GBPUSD", "EURGBP"]: display_precision = 5

    # Return raw rate for calculation, display rate, and data pair key
    return rate_value, round(rate_value, display_precision), data_pair


def simulate_enhanced_prediction(pair, amount):
    """
    Simulates a more 'intelligent' prediction using enhanced mock data.
    Calculates potential savings based on the prediction.
    Uses cautious wording.

    NOTE: This is still a SIMULATION. Real-world implementation requires
    live data feeds, actual ML models, robust infrastructure, backtesting, etc.
    """
    raw_rate, display_rate, data_pair_key = get_current_rate_and_details(pair)

    if raw_rate is None:
         return {"error": f"Rate for {pair} not available in mock data."}

    # --- Get Underlying Data ---
    if data_pair_key is None: # Should not happen if raw_rate is not None
        return {"error": f"Internal Error: Historical data key for {pair} not found."}
    history = MOCK_FX_HISTORY.get(data_pair_key, [])
    is_inverse_data_used = (pair != data_pair_key)
    # --- End Underlying Data ---

    # Check for sufficient history for full analysis
    MIN_HISTORY = 24 # Need at least 24 hours for better analysis
    if len(history) < MIN_HISTORY:
        return {
            "recommendation": "stable",
            "message": "Displaying current rate. Insufficient historical data for detailed AI analysis.",
            "current_rate": display_rate, # Use display rate for consistency
            "pair": pair,
            "details": {},
            "savings": None # No savings calculation possible
        }

    # --- 1. Enhanced Trend & Volatility Analysis (using underlying data) ---
    rates_recent = [r for _, r in history[-MIN_HISTORY:]]
    trend_factor = 0.0; volatility_factor = 0.0; trend_desc = "uncertain"; trend_icon = "❓"
    try:
        if len(rates_recent) >= 5: # Need a few points
             time_axis = np.arange(len(rates_recent))
             slope, _ = np.polyfit(time_axis, rates_recent, 1)
             avg_rate = np.mean(rates_recent)
             if avg_rate != 0:
                  # Relative trend strength over the period
                 trend_factor = (slope * len(rates_recent)) / avg_rate
                 # Simple volatility measure (std dev relative to mean)
                 volatility_factor = np.std(rates_recent) / avg_rate

                 # More nuanced trend description
                 if trend_factor > 0.002: trend_desc = "strong upward"; trend_icon = "🚀"
                 elif trend_factor > 0.0005: trend_desc = "modest upward"; trend_icon = "📈"
                 elif trend_factor < -0.002: trend_desc = "strong downward"; trend_icon = "💥"
                 elif trend_factor < -0.0005: trend_desc = "modest downward"; trend_icon = "📉"
                 else: trend_desc = "flat/ranging"; trend_icon = "↔️"

                 if volatility_factor > 0.005: # High recent volatility?
                     trend_desc += " (volatile)"
                     trend_icon = "🌊"

    except Exception as e:
        print(f"Trend/Vol calculation error for {data_pair_key}: {e}")
        trend_desc = "error"; trend_icon = "⚠️"

    # Adjust for requested pair if inverse data was used
    trend_summary_desc = trend_desc; trend_summary_icon = trend_icon
    if is_inverse_data_used: # Invert trend description and factor
        trend_factor = -trend_factor
        if "upward" in trend_desc: trend_summary_desc = trend_desc.replace("upward","downward"); trend_summary_icon="📉" # Simplified swap
        elif "downward" in trend_desc: trend_summary_desc = trend_desc.replace("downward","upward"); trend_summary_icon="📈"
        # Note: Strong/Modest swap isn't perfect logic here, refine if needed
        if "🚀" in trend_icon: trend_summary_icon = "💥"
        elif "📈" in trend_icon: trend_summary_icon = "📉"
        elif "💥" in trend_icon: trend_summary_icon = "🚀"
        elif "📉" in trend_icon: trend_summary_icon = "📈"

    trend_summary = f"{trend_summary_icon} Recent ({MIN_HISTORY}h) price action shows {trend_summary_desc} for {pair}."
    if "error" in trend_desc: trend_summary = f"{trend_summary_icon} Error analyzing recent price action."
    # --- End Trend Analysis ---


    # --- 2. Enhanced Fundamental Scoring ---
    ccy1, ccy2 = pair[:3], pair[3:]
    fund_ccy1 = FUNDAMENTAL_DATA.get(ccy1); fund_ccy2 = FUNDAMENTAL_DATA.get(ccy2)
    fundamental_score = 0; fundamental_drivers = []
    if fund_ccy1 and fund_ccy2:
        # Interest Rate Differential (higher weight)
        rate_diff = fund_ccy1["interest_rate"] - fund_ccy2["interest_rate"]
        fundamental_score += rate_diff * 0.3
        if abs(rate_diff) > 0.5: fundamental_drivers.append(f"Rate diff ({rate_diff:.2f}%)")

        # Growth Differential
        gdp_diff = fund_ccy1["gdp_growth"] - fund_ccy2["gdp_growth"]
        fundamental_score += gdp_diff * 0.15
        if abs(gdp_diff) > 1.0: fundamental_drivers.append(f"Growth diff ({gdp_diff:.1f}%)")

        # Inflation Differential (inverse effect - higher inflation often weakens currency short-term)
        inflation_diff = fund_ccy1["inflation"] - fund_ccy2["inflation"]
        fundamental_score -= inflation_diff * 0.1
        # Unemployment Differential (inverse effect - lower unemployment often strengthens)
        unemp_diff = fund_ccy1["unemployment"] - fund_ccy2["unemployment"]
        fundamental_score -= unemp_diff * 0.05

        # Outlook Difference
        outlook_map = {"strong": 1, "neutral": 0, "weak": -1, "mixed": 0}
        outlook_diff = outlook_map.get(fund_ccy1["outlook"], 0) - outlook_map.get(fund_ccy2["outlook"], 0)
        fundamental_score += outlook_diff * 0.2
        if abs(outlook_diff) > 0: fundamental_drivers.append(f"{ccy1} outlook vs {ccy2}")

    fundamental_summary = f"🏦 Economic factors appear {'slightly positive' if fundamental_score > 0.1 else 'slightly negative' if fundamental_score < -0.1 else 'balanced'} for {ccy1} vs {ccy2}."
    if fundamental_drivers:
        fundamental_summary += f" Key drivers: {', '.join(fundamental_drivers[:2])}."
    # --- End Fundamental Analysis ---

    # --- 3. Enhanced News/Sentiment Analysis ---
    news_score = 0; impactful_news = []
    for news in MOCK_NEWS:
        affects_ccy1 = ccy1 in news["affects"]
        affects_ccy2 = ccy2 in news["affects"]
        if affects_ccy1 or affects_ccy2:
            score_effect = news["sentiment"] * news["impact"] # Factor in impact
            if affects_ccy1 and affects_ccy2: score_effect *= 0.3 # Muted effect if both impacted
            elif affects_ccy2: score_effect = -score_effect # Invert if only target ccy affected
            news_score += score_effect
            if abs(score_effect) > 0.1: # Only list impactful news
                 impactful_news.append(f"\"{news['headline'][:40]}...\" ({'+' if score_effect > 0 else ''}{score_effect:.1f})")

    news_summary = "📰 Recent news flow seems neutral or mixed."
    if news_score > 0.15: news_summary = f"📰 Recent news appears somewhat supportive for {ccy1}."
    elif news_score < -0.15: news_summary = f"📰 Recent news appears somewhat challenging for {ccy1}."
    if impactful_news: news_summary += f" Impactful: {'; '.join(impactful_news[:1])}" # Show one example
    # --- End News Analysis ---

    # --- 4. Technical Indicator Input (Simple) ---
    tech_score = 0
    indicators = TECHNICAL_INDICATORS.get(data_pair_key) # Use underlying pair for tech analysis
    if indicators:
        # Example: RSI divergence from price trend, MACD signal
        if indicators["rsi"] > 65 and trend_factor < -0.0005: tech_score -= 0.1 # Bearish divergence?
        if indicators["rsi"] < 35 and trend_factor > 0.0005: tech_score += 0.1 # Bullish divergence?
        tech_score += indicators["macd_signal"] * 0.05 # Small weight for MACD cross

        # Adjust score if inverse data was used
        if is_inverse_data_used: tech_score = -tech_score
    # --- End Technical Input ---


    # --- 5. Combine Factors & Generate Recommendation ---
    # Weights: Fundamentals (medium term) > Trend (short term) > News (fleeting) > Technical (confirmation)
    # Note: These weights are arbitrary for simulation!
    final_score = (fundamental_score * 0.4) + (trend_factor * 50 * 0.3) + (news_score * 0.2) + (tech_score * 0.1)

    # Dynamic Thresholds based on Volatility? (Simple example)
    base_threshold = 0.30
    vol_adj_factor = 1 + (FUNDAMENTAL_DATA.get(ccy1,{}).get("vol_index", 15) + FUNDAMENTAL_DATA.get(ccy2,{}).get("vol_index", 15)) / 50.0 # Higher vol -> wider threshold
    action_threshold = base_threshold * vol_adj_factor

    recommendation = "stable"
    rec_explanation_core = ""

    # Determine recommendation
    if final_score > action_threshold:
        recommendation = "wait"; rec_explanation_core = f"Analysis suggests a potential for the {pair} rate to become more favorable soon."
    elif final_score < -action_threshold:
        recommendation = "send_now"; rec_explanation_core = f"Analysis suggests the current {pair} rate might be favorable compared to the near future."
    elif abs(final_score) < (action_threshold / 2.0): # Clear stable zone
        recommendation = "stable"; rec_explanation_core = f"The {pair} rate appears relatively stable based on current analysis."
    else: # Uncertain zone
        recommendation = "uncertain"; rec_explanation_core = f"Signals for {pair} are mixed; consider your risk tolerance."

    final_message = f"AI Advice: {rec_explanation_core} (Based on simulated data; predictions not guaranteed)."

    # --- 6. Calculate Potential Savings ---
    savings_info = None
    if recommendation in ["wait", "send_now"] and amount > 0:
        # Estimate potential rate change based on score strength and volatility
        # VERY simplified simulation - real model would provide confidence intervals or predicted range
        potential_rate_change_pct = abs(final_score) * 0.001 * (1 + volatility_factor) # Tiny % change based on score/vol
        potential_rate_change_pct = min(potential_rate_change_pct, 0.005) # Cap potential change

        if recommendation == "wait": # Expect rate for CCY1/CCY2 to go UP
            potential_future_rate = raw_rate * (1 + potential_rate_change_pct)
            potential_received_amount = amount * potential_future_rate
            current_received_amount = amount * raw_rate
            savings = potential_received_amount - current_received_amount
            if savings > 0.01: # Only show meaningful savings
                 savings_info = {
                     "amount": round(savings, 2),
                     "currency": ccy2,
                     "percent": round((potential_future_rate / raw_rate - 1) * 100, 2),
                     "message": f"Waiting could potentially save ~{round(savings, 2)} {ccy2} (rate improvement of ~{round((potential_future_rate / raw_rate - 1) * 100, 2)}%)."
                 }
        elif recommendation == "send_now": # Expect rate for CCY1/CCY2 to go DOWN
            potential_future_rate = raw_rate * (1 - potential_rate_change_pct)
            potential_received_amount = amount * potential_future_rate
            current_received_amount = amount * raw_rate
            savings = current_received_amount - potential_received_amount # Savings = loss avoided
            if savings > 0.01:
                 savings_info = {
                     "amount": round(savings, 2),
                     "currency": ccy2,
                     "percent": round((1 - potential_future_rate / raw_rate) * 100, 2),
                     "message": f"Sending now might avoid a potential loss of ~{round(savings, 2)} {ccy2} (rate worsening of ~{round((1 - potential_future_rate / raw_rate) * 100, 2)}%)."
                 }

    # --- End Savings Calculation ---

    return {
        "recommendation": recommendation,
        "message": final_message,
        "current_rate": display_rate, # Use display rate in final output
        "pair": pair,
        "details": {
            "trend_summary": trend_summary,
            "fundamental_summary": fundamental_summary,
            "news_summary": news_summary,
            "score": round(final_score, 3) # Keep score for potential debugging/display
        },
        "savings": savings_info # Add savings info
    }
# --- End Prediction Function ---


# --- Routes ---
@app.route('/')
def index():
    """ Renders the main transfer UI page. """
    preferred_order = ["USD", "EUR", "GBP", "INR", "JPY"]
    sorted_currencies = [c for c in preferred_order if c in SUPPORTED_CURRENCIES] + \
                        [c for c in SUPPORTED_CURRENCIES if c not in preferred_order]
    return render_template('index.html', currencies=sorted_currencies)

@app.route('/api/fx_prediction/<pair>/<amount_str>') # Add amount to API route
def get_fx_prediction_api(pair, amount_str):
    """ API endpoint to get the simulated FX prediction, rate, and savings. """
    pair = pair.upper()
    try:
        amount = float(amount_str)
    except ValueError:
        return jsonify({"error": "Invalid amount format."}), 400

    if len(pair) != 6: return jsonify({"error": f"Invalid currency pair format: {pair}"}), 400
    ccy1, ccy2 = pair[:3], pair[3:]
    if not (ccy1 in SUPPORTED_CURRENCIES and ccy2 in SUPPORTED_CURRENCIES and ccy1 != ccy2):
         return jsonify({"error": f"Invalid or unsupported currency pair: {pair}"}), 400

    # Pass amount to prediction function
    prediction = simulate_enhanced_prediction(pair, amount)

    if "error" in prediction and "not available" in prediction["error"]:
         return jsonify(prediction), 404 # Rate Not Found
    elif "error" in prediction:
        return jsonify(prediction), 500 # Internal Server Error
    else:
         return jsonify(prediction) # 200 OK

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)