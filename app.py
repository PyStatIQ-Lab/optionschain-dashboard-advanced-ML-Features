import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="PyStatIQ Options Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    .prediction-card {
        background-color: #f1f8fe;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .tabs {
        margin-bottom: 20px;
    }
    .trade-recommendation {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        color: #000;
    }
    .trade-recommendation.sell {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
    }
    .strike-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #40404f;
    }
    .regime-low {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }
    .regime-normal {
        background-color: #fff3e0;
        border-left: 5px solid #fb8c00;
    }
    .regime-high {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
    }
    .ml-signal {
        font-weight: bold;
        padding: 3px 6px;
        border-radius: 4px;
    }
    .signal-buy {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .signal-sell {
        background-color: #ffebee;
        color: #c62828;
    }
    .signal-neutral {
        background-color: #e3f2fd;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
MARKET_DATA_URL = "https://service.upstox.com/market-data-api/v2/open/quote"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# Constants
RISK_FREE_RATE = 0.05  # 5% risk-free rate for calculations
DAYS_IN_YEAR = 365

# Fetch data from API
@st.cache_data(ttl=300)
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="03-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

# Fetch live Nifty price
@st.cache_data(ttl=60)
def fetch_nifty_price():
    url = f"{MARKET_DATA_URL}?i=NSE_INDEX|Nifty%2050"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        return data['data']['lastPrice']
    else:
        st.error(f"Failed to fetch Nifty price: {response.status_code} - {response.text}")
        return None

# Process raw API data
def process_options_data(raw_data, spot_price):
    if not raw_data or 'data' not in raw_data:
        return None
    
    strike_map = raw_data['data']['strategyChainData']['strikeMap']
    processed_data = []
    
    for strike, data in strike_map.items():
        call_data = data.get('callOptionData', {})
        put_data = data.get('putOptionData', {})
        
        # Market data
        call_market = call_data.get('marketData', {})
        put_market = put_data.get('marketData', {})
        
        # Analytics data
        call_analytics = call_data.get('analytics', {})
        put_analytics = put_data.get('analytics', {})
        
        strike_float = float(strike)
        
        processed_data.append({
            'strike': strike_float,
            'pcr': data.get('pcr', 0),
            
            # Moneyness
            'call_moneyness': 'ITM' if strike_float < spot_price else ('ATM' if strike_float == spot_price else 'OTM'),
            'put_moneyness': 'ITM' if strike_float > spot_price else ('ATM' if strike_float == spot_price else 'OTM'),
            
            # Call data
            'call_ltp': call_market.get('ltp', 0),
            'call_bid': call_market.get('bidPrice', 0),
            'call_ask': call_market.get('askPrice', 0),
            'call_volume': call_market.get('volume', 0),
            'call_oi': call_market.get('oi', 0),
            'call_prev_oi': call_market.get('prevOi', 0),
            'call_oi_change': call_market.get('oi', 0) - call_market.get('prevOi', 0),
            'call_iv': call_analytics.get('iv', 0),
            'call_delta': call_analytics.get('delta', 0),
            'call_gamma': call_analytics.get('gamma', 0),
            'call_theta': call_analytics.get('theta', 0),
            'call_vega': call_analytics.get('vega', 0),
            
            # Put data
            'put_ltp': put_market.get('ltp', 0),
            'put_bid': put_market.get('bidPrice', 0),
            'put_ask': put_market.get('askPrice', 0),
            'put_volume': put_market.get('volume', 0),
            'put_oi': put_market.get('oi', 0),
            'put_prev_oi': put_market.get('prevOi', 0),
            'put_oi_change': put_market.get('oi', 0) - put_market.get('prevOi', 0),
            'put_iv': put_analytics.get('iv', 0),
            'put_delta': put_analytics.get('delta', 0),
            'put_gamma': put_analytics.get('gamma', 0),
            'put_theta': put_analytics.get('theta', 0),
            'put_vega': put_analytics.get('vega', 0),
        })
    
    return pd.DataFrame(processed_data)

# Get top ITM/OTM strikes
def get_top_strikes(df, spot_price, n=5):
    call_itm = df[df['strike'] < spot_price].sort_values('strike', ascending=False).head(n)
    call_otm = df[df['strike'] > spot_price].sort_values('strike', ascending=True).head(n)
    put_itm = df[df['strike'] > spot_price].sort_values('strike', ascending=True).head(n)
    put_otm = df[df['strike'] < spot_price].sort_values('strike', ascending=False).head(n)
    
    return {
        'call_itm': call_itm,
        'call_otm': call_otm,
        'put_itm': put_itm,
        'put_otm': put_otm
    }

# Machine Learning Model for Trade Signals
class OptionTradeModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        
    def prepare_features(self, df, spot_price):
        features.append([
                row['call_iv'] - row['put_iv'],  # IV Skew
                row['call_oi_change'],
                row['put_oi_change'],
                (row['call_ask'] - row['call_bid']) / row['call_ltp'] if row['call_ltp'] > 0 else 0,  # Call spread %
                (row['put_ask'] - row['put_bid']) / row['put_ltp'] if row['put_ltp'] > 0 else 0,  # Put spread %
                row['call_volume'] / (df['call_volume'].max() + 1e-6),  # Normalized volume
                row['put_volume'] / (df['put_volume'].max() + 1e-6),
                (spot_price - row['strike']) / spot_price if row['call_moneyness'] == 'OTM' else 0,  # % OTM for calls
                (row['strike'] - spot_price) / spot_price if row['put_moneyness'] == 'OTM' else 0,  # % OTM for puts
                row['call_gamma'],
                row['put_gamma']
            ])
        return np.array(features)
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
        
    def predict(self, X):
        if not self.trained:
            return np.zeros(X.shape[0])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]  # Probability of being a good buy
    
    def feature_importance(self):
        if not self.trained:
            return None
        return self.model.feature_importances_

# Market Regime Detection using HMM
class MarketRegimeDetector:
    def __init__(self, n_states=3):
        self.model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
        self.regime_map = {0: "Low Volatility", 1: "Normal", 2: "High Volatility"}
        
    def fit(self, historical_data):
        # historical_data should be a DataFrame with columns: ['iv', 'volume', 'returns']
        self.model.fit(historical_data[['iv', 'volume', 'returns']].values.reshape(-1, 3))
        
    def predict_regime(self, current_data):
        # current_data should be a dict with: {'iv': x, 'volume': y, 'returns': z}
        regime = self.model.predict(np.array([[current_data['iv'], current_data['volume'], current_data['returns']]]))
        return self.regime_map[regime[0]]
    
    def get_expected_move(self, current_iv, days_to_expiry):
        # Calculate expected move based on IV
        return current_iv * np.sqrt(days_to_expiry / DAYS_IN_YEAR)

# Arbitrage Detection
def detect_arbitrage_opportunities(df, spot_price, risk_free_rate, days_to_expiry):
    arbitrage_ops = []
    for _, row in df.iterrows():
        strike = row['strike']
        call_price = row['call_ltp']
        put_price = row['put_ltp']
        
        # Put-call parity: C - P = S - K*e^(-rT)
        theoretical_diff = spot_price - strike * np.exp(-risk_free_rate * days_to_expiry / DAYS_IN_YEAR)
        actual_diff = call_price - put_price
        
        # Arbitrage opportunities
        if abs(actual_diff - theoretical_diff) > max(5, 0.05 * spot_price):  # Threshold
            arbitrage_ops.append({
                'strike': strike,
                'call_price': call_price,
                'put_price': put_price,
                'theoretical_diff': theoretical_diff,
                'actual_diff': actual_diff,
                'arbitrage_amount': abs(actual_diff - theoretical_diff),
                'direction': 'Buy Put + Sell Call' if actual_diff > theoretical_diff else 'Buy Call + Sell Put'
            })
    
    return pd.DataFrame(arbitrage_ops).sort_values('arbitrage_amount', ascending=False)

# Probability of Profit Calculator
def calculate_probability_of_profit(option_type, strike, premium, spot_price, iv, days_to_expiry):
    if option_type == 'call':
        break_even = strike + premium
        expected_move = spot_price * iv * np.sqrt(days_to_expiry / DAYS_IN_YEAR)
        z_score = (break_even - spot_price) / (spot_price * iv * np.sqrt(days_to_expiry / DAYS_IN_YEAR))
    else:  # put
        break_even = strike - premium
        expected_move = spot_price * iv * np.sqrt(days_to_expiry / DAYS_IN_YEAR)
        z_score = (spot_price - break_even) / (spot_price * iv * np.sqrt(days_to_expiry / DAYS_IN_YEAR))
    
    return 1 - norm.cdf(z_score)

# Smart Money Flow Detection
def detect_smart_money_flows(df, spot_price, volume_threshold, oi_change_threshold):
    smart_money = []
    
    # Criteria for institutional activity
    for _, row in df.iterrows():
        # Call side smart money detection
        if (row['call_volume'] > volume_threshold and 
            row['call_oi_change'] > oi_change_threshold and 
            row['call_iv'] < df['call_iv'].median()):
            smart_money.append({
                'strike': row['strike'],
                'side': 'Call',
                'volume': row['call_volume'],
                'oi_change': row['call_oi_change'],
                'iv': row['call_iv'],
                'signal': 'Institutional Buying' if row['call_ltp'] > (row['call_bid'] + row['call_ask']) / 2 else 'Institutional Selling'
            })
        
        # Put side smart money detection
        if (row['put_volume'] > volume_threshold and 
            row['put_oi_change'] > oi_change_threshold and 
            row['put_iv'] < df['put_iv'].median()):
            smart_money.append({
                'strike': row['strike'],
                'side': 'Put',
                'volume': row['put_volume'],
                'oi_change': row['put_oi_change'],
                'iv': row['put_iv'],
                'signal': 'Institutional Buying' if row['put_ltp'] > (row['put_bid'] + row['put_ask']) / 2 else 'Institutional Selling'
            })
    
    return pd.DataFrame(smart_money).sort_values(['volume', 'oi_change'], ascending=False)

# Generate trade recommendations with ML signals
def generate_trade_recommendations(df, spot_price, model, days_to_expiry=1):
    recommendations = []
    
    # Prepare features for ML model
    def prepare_features(self, df, spot_price):
    features = []  # Initialize the features list
    for _, row in df.iterrows():
        features.append([
            row['call_iv'] - row['put_iv'],  # IV Skew
            row['call_oi_change'],
            row['put_oi_change'],
            (row['call_ask'] - row['call_bid']) / row['call_ltp'] if row['call_ltp'] > 0 else 0,  # Call spread %
            (row['put_ask'] - row['put_bid']) / row['put_ltp'] if row['put_ltp'] > 0 else 0,  # Put spread %
            row['call_volume'] / (df['call_volume'].max() + 1e-6),  # Normalized volume
            row['put_volume'] / (df['put_volume'].max() + 1e-6),
            (spot_price - row['strike']) / spot_price if row['call_moneyness'] == 'OTM' else 0,  # % OTM for calls
            (row['strike'] - spot_price) / spot_price if row['put_moneyness'] == 'OTM' else 0,  # % OTM for puts
            row['call_gamma'],
            row['put_gamma']
        ])
    return np.array(features)
    
    # Get top ML-scored calls and puts
    top_calls = df[df['call_moneyness'] == 'OTM'].sort_values('ml_score', ascending=False).head(3)
    top_puts = df[df['put_moneyness'] == 'OTM'].sort_values('ml_score', ascending=False).head(3)
    
    # Generate recommendations with ML signals
    for _, row in top_calls.iterrows():
        pop = calculate_probability_of_profit('call', row['strike'], row['call_ltp'], spot_price, row['call_iv'], days_to_expiry)
        recommendations.append({
            'type': 'BUY CALL',
            'strike': row['strike'],
            'premium': row['call_ltp'],
            'iv': row['call_iv'],
            'oi_change': row['call_oi_change'],
            'ml_score': row['ml_score'],
            'pop': pop,
            'reason': f"ML Score: {row['ml_score']:.2f}, Prob of Profit: {pop:.1%}"
        })
    
    for _, row in top_puts.iterrows():
        pop = calculate_probability_of_profit('put', row['strike'], row['put_ltp'], spot_price, row['put_iv'], days_to_expiry)
        recommendations.append({
            'type': 'BUY PUT',
            'strike': row['strike'],
            'premium': row['put_ltp'],
            'iv': row['put_iv'],
            'oi_change': row['put_oi_change'],
            'ml_score': row['ml_score'],
            'pop': pop,
            'reason': f"ML Score: {row['ml_score']:.2f}, Prob of Profit: {pop:.1%}"
        })
    
    # Find overpriced options to sell (high IV percentile)
    high_iv_calls = df[(df['call_moneyness'] == 'ITM') & 
                      (df['call_iv'] > df['call_iv'].quantile(0.75))].sort_values('call_iv', ascending=False).head(2)
    
    for _, row in high_iv_calls.iterrows():
        pop = calculate_probability_of_profit('call', row['strike'], row['call_ltp'], spot_price, row['call_iv'], days_to_expiry)
        recommendations.append({
            'type': 'SELL CALL',
            'strike': row['strike'],
            'premium': row['call_ltp'],
            'iv': row['call_iv'],
            'oi_change': row['call_oi_change'],
            'ml_score': 1 - row['ml_score'],
            'pop': pop,
            'reason': f"High IV ({row['call_iv']:.1f}%), Prob of Profit: {pop:.1%}"
        })
    
    high_iv_puts = df[(df['put_moneyness'] == 'ITM') & 
                     (df['put_iv'] > df['put_iv'].quantile(0.75))].sort_values('put_iv', ascending=False).head(2)
    
    for _, row in high_iv_puts.iterrows():
        pop = calculate_probability_of_profit('put', row['strike'], row['put_ltp'], spot_price, row['put_iv'], days_to_expiry)
        recommendations.append({
            'type': 'SELL PUT',
            'strike': row['strike'],
            'premium': row['put_ltp'],
            'iv': row['put_iv'],
            'oi_change': row['put_oi_change'],
            'ml_score': 1 - row['ml_score'],
            'pop': pop,
            'reason': f"High IV ({row['put_iv']:.1f}%), Prob of Profit: {pop:.1%}"
        })
    
    return recommendations, df

# Main App
def main():
    st.markdown("<div class='header'><h1>📊 PyStatIQ Options Chain Dashboard</h1></div>", unsafe_allow_html=True)
    
    # Initialize models
    trade_model = OptionTradeModel()
    regime_detector = MarketRegimeDetector()
    
    # Simulate training data (in practice, you'd use historical data)
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, 100)  # Binary labels
    trade_model.train(X_train, y_train)
    
    # Simulate regime detection training
    historical_data = pd.DataFrame({
        'iv': np.random.uniform(10, 50, 100),
        'volume': np.random.uniform(1e6, 1e7, 100),
        'returns': np.random.normal(0, 0.01, 100)
    })
    regime_detector.fit(historical_data)
    
    # Fetch spot price
    spot_price = fetch_nifty_price()
    if spot_price is None:
        st.error("Failed to fetch Nifty spot price. Using default value.")
        spot_price = 22000  # Default fallback
    
    # Sidebar controls
    with st.sidebar:
        st.header("Filters")
        asset_key = st.selectbox(
            "Underlying Asset",
            ["NSE_INDEX|Nifty 50", "NSE_INDEX|Bank Nifty"],
            index=0
        )
        
        expiry_date = st.date_input(
            "Expiry Date",
            datetime.strptime("03-04-2025", "%d-%m-%Y")
        ).strftime("%d-%m-%Y")
        
        days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=30, value=1)
        
        st.markdown("---")
        st.markdown(f"**Current Nifty Spot Price: {spot_price:,.2f}**")
        
        st.markdown("---")
        st.markdown("**Analysis Settings**")
        volume_threshold = st.number_input("High Volume Threshold", value=5000000)
        oi_change_threshold = st.number_input("Significant OI Change", value=1000000)
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("Advanced options analytics dashboard with ML signals, risk metrics, and smart money detection.")
    
    # Fetch and process data
    with st.spinner("Fetching live options data..."):
        raw_data = fetch_options_data(asset_key, expiry_date)
    
    if raw_data is None:
        st.error("Failed to load data. Please try again later.")
        return
    
    df = process_options_data(raw_data, spot_price)
    if df is None or df.empty:
        st.error("No data available for the selected parameters.")
        return
    
    # Get top strikes
    top_strikes = get_top_strikes(df, spot_price)
    
    # Default strike selection (ATM)
    atm_strike = df.iloc[(df['strike'] - spot_price).abs().argsort()[:1]]['strike'].values[0]
    
    # Market Regime Detection
    current_volatility = df['call_iv'].mean()  # Using average call IV as proxy
    current_volume = df['call_volume'].sum() + df['put_volume'].sum()
    current_return = 0  # Would normally get from historical data
    
    regime = regime_detector.predict_regime({
        'iv': current_volatility,
        'volume': current_volume,
        'returns': current_return
    })
    
    expected_move_pct = regime_detector.get_expected_move(current_volatility, days_to_expiry)
    expected_move_points = spot_price * expected_move_pct
    
    # Main columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Call OI**")
        total_call_oi = df['call_oi'].sum()
        st.markdown(f"<h2>{total_call_oi:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Put OI**")
        total_put_oi = df['put_oi'].sum()
        st.markdown(f"<h2>{total_put_oi:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Strike price selector
        selected_strike = st.selectbox(
            "Select Strike Price",
            df['strike'].unique(),
            index=int(np.where(df['strike'].unique() == atm_strike)[0][0])
        )
        
        # Market regime display
        regime_class = "regime-low" if "Low" in regime else ("regime-high" if "High" in regime else "regime-normal")
        st.markdown(f"""
            <div class='metric-card {regime_class}'>
                <h3>Market Regime: {regime}</h3>
                <p>Expected Move: ±{expected_move_pct:.2%} (±{expected_move_points:.0f} points)</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Call OI Change**")
        call_oi_change = df[df['strike'] == selected_strike]['call_oi_change'].values[0]
        change_color = "positive" if call_oi_change > 0 else "negative"
        st.markdown(f"<h2 class='{change_color}'>{call_oi_change:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Put OI Change**")
        put_oi_change = df[df['strike'] == selected_strike]['put_oi_change'].values[0]
        change_color = "positive" if put_oi_change > 0 else "negative"
        st.markdown(f"<h2 class='{change_color}'>{put_oi_change:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Top Strikes Section
    st.markdown("### Top ITM/OTM Strike Prices")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Top ITM Call Strikes**")
        for _, row in top_strikes['call_itm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['call_ltp']:.2f})<br>
                    OI: {row['call_oi']:,} (Δ: {row['call_oi_change']:,})<br>
                    IV: {row['call_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Top OTM Call Strikes**")
        for _, row in top_strikes['call_otm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['call_ltp']:.2f})<br>
                    OI: {row['call_oi']:,} (Δ: {row['call_oi_change']:,})<br>
                    IV: {row['call_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Top ITM Put Strikes**")
        for _, row in top_strikes['put_itm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['put_ltp']:.2f})<br>
                    OI: {row['put_oi']:,} (Δ: {row['put_oi_change']:,})<br>
                    IV: {row['put_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Top OTM Put Strikes**")
        for _, row in top_strikes['put_otm'].iterrows():
            st.markdown(f"""
                <div class='strike-card'>
                    <b>{row['strike']:.0f}</b> (LTP: {row['put_ltp']:.2f})<br>
                    OI: {row['put_oi']:,} (Δ: {row['put_oi_change']:,})<br>
                    IV: {row['put_iv']:.1f}%
                </div>
            """, unsafe_allow_html=True)
    
    # Trade Recommendations with ML
    st.markdown("### AI-Powered Trade Recommendations")
    recommendations, df_with_scores = generate_trade_recommendations(df, spot_price, trade_model, days_to_expiry)
    
    if recommendations:
        for rec in recommendations:
            is_sell = 'SELL' in rec['type']
            ml_score_class = "signal-buy" if rec['ml_score'] > 0.7 else ("signal-sell" if rec['ml_score'] < 0.3 else "signal-neutral")
            
            st.markdown(f"""
                <div class='trade-recommendation{' sell' if is_sell else ''}'>
                    <h4>{rec['type']} @ {rec['strike']:.0f} 
                        <span class='ml-signal {ml_score_class}'>ML Score: {rec['ml_score']:.2f}</span>
                        <span style='float:right;'>Prob Profit: {rec['pop']:.1%}</span>
                    </h4>
                    <p>
                        Premium: {rec['premium']:.2f} | IV: {rec['iv']:.1f}% | OI Change: {rec['oi_change']:,}<br>
                        <b>Reason:</b> {rec['reason']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No strong trade recommendations based on current market conditions")
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["Strike Analysis", "OI/Volume Trends", "Advanced Analytics", "Predictive Models"])
    
    with tab1:
        st.markdown(f"### Detailed Analysis for Strike: {selected_strike}")
        
        # Get selected strike data
        strike_data = df[df['strike'] == selected_strike].iloc[0]
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['LTP', 'Bid', 'Ask', 'Volume', 'OI', 'OI Change', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega'],
            'Call': [
                strike_data['call_ltp'],
                strike_data['call_bid'],
                strike_data['call_ask'],
                strike_data['call_volume'],
                strike_data['call_oi'],
                strike_data['call_oi_change'],
                strike_data['call_iv'],
                strike_data['call_delta'],
                strike_data['call_gamma'],
                strike_data['call_theta'],
                strike_data['call_vega']
            ],
            'Put': [
                strike_data['put_ltp'],
                strike_data['put_bid'],
                strike_data['put_ask'],
                strike_data['put_volume'],
                strike_data['put_oi'],
                strike_data['put_oi_change'],
                strike_data['put_iv'],
                strike_data['put_delta'],
                strike_data['put_gamma'],
                strike_data['put_theta'],
                strike_data['put_vega']
            ]
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Call': '{:,.2f}',
                'Put': '{:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        st.markdown("### Open Interest & Volume Trends")
        
        # Nearby strikes
        all_strikes = sorted(df['strike'].unique())
        current_idx = all_strikes.index(selected_strike)
        nearby_strikes = all_strikes[max(0, current_idx-5):min(len(all_strikes), current_idx+6)]
        nearby_df = df[df['strike'].isin(nearby_strikes)]
        
        # OI Change plot
        fig = px.bar(
            nearby_df,
            x='strike',
            y=['call_oi_change', 'put_oi_change'],
            barmode='group',
            title=f'OI Changes Around {selected_strike}',
            labels={'value': 'OI Change', 'strike': 'Strike Price'},
            color_discrete_map={'call_oi_change': '#3498db', 'put_oi_change': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume plot
        fig = px.bar(
            nearby_df,
            x='strike',
            y=['call_volume', 'put_volume'],
            barmode='group',
            title=f'Volume Around {selected_strike}',
            labels={'value': 'Volume', 'strike': 'Strike Price'},
            color_discrete_map={'call_volume': '#3498db', 'put_volume': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Arbitrage Opportunities
            st.markdown("#### Arbitrage Opportunities")
            arbitrage_ops = detect_arbitrage_opportunities(df, spot_price, RISK_FREE_RATE, days_to_expiry)
            
            if not arbitrage_ops.empty:
                st.dataframe(
                    arbitrage_ops.style.format({
                        'call_price': '{:.2f}',
                        'put_price': '{:.2f}',
                        'theoretical_diff': '{:.2f}',
                        'actual_diff': '{:.2f}',
                        'arbitrage_amount': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No significant arbitrage opportunities detected")
            
            # Smart Money Flows
            st.markdown("#### Smart Money Flow Detection")
            smart_money = detect_smart_money_flows(df, spot_price, volume_threshold, oi_change_threshold)
            
            if not smart_money.empty:
                st.dataframe(
                    smart_money.style.format({
                        'volume': '{:,}',
                        'oi_change': '{:,}',
                        'iv': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No unusual smart money activity detected")
        
        with col2:
            # IV Skew Analysis
            st.markdown("#### IV Skew Analysis")
            fig = px.line(
                df,
                x='strike',
                y=['call_iv', 'put_iv'],
                title='Implied Volatility Skew',
                labels={'value': 'IV (%)', 'strike': 'Strike Price'},
                color_discrete_map={'call_iv': '#3498db', 'put_iv': '#e74c3c'}
            )
            fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            # Greeks Exposure
            st.markdown("#### Greeks Exposure")
            fig = px.line(
                df,
                x='strike',
                y=['call_gamma', 'put_gamma', 'call_theta', 'put_theta'],
                title='Gamma and Theta Exposure',
                labels={'value': 'Value', 'strike': 'Strike Price'},
                color_discrete_map={
                    'call_gamma': '#3498db', 
                    'put_gamma': '#e74c3c',
                    'call_theta': '#2ecc71',
                    'put_theta': '#f39c12'
                }
            )
            fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Predictive Models & ML Signals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ML Model Feature Importance
            st.markdown("#### Model Feature Importance")
            feature_importance = trade_model.feature_importance()
            if feature_importance is not None:
                features = [
                    'IV Skew', 'Call OI Change', 'Put OI Change', 
                    'Call Spread %', 'Put Spread %', 'Call Volume', 
                    'Put Volume', '% OTM Call', '% OTM Put',
                    'Call Gamma', 'Put Gamma'
                ]
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance in Trade Model'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model not trained or feature importance not available")
            
            # ML Scores Distribution
            st.markdown("#### ML Scores Distribution")
            if 'ml_score' in df_with_scores.columns:
                fig = px.histogram(
                    df_with_scores,
                    x='ml_score',
                    nbins=20,
                    title='Distribution of ML Scores Across Strikes',
                    labels={'ml_score': 'ML Score (0-1)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top ML-Scored Calls
            st.markdown("#### Top ML-Scored Calls")
            top_calls = df_with_scores[df_with_scores['call_moneyness'] == 'OTM'].sort_values('ml_score', ascending=False).head(10)
            st.dataframe(
                top_calls[['strike', 'call_ltp', 'call_iv', 'call_oi_change', 'ml_score']]
                .rename(columns={
                    'strike': 'Strike',
                    'call_ltp': 'Price',
                    'call_iv': 'IV',
                    'call_oi_change': 'OI Change',
                    'ml_score': 'ML Score'
                })
                .style.format({
                    'Price': '{:.2f}',
                    'IV': '{:.1f}%',
                    'OI Change': '{:,}',
                    'ML Score': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Top ML-Scored Puts
            st.markdown("#### Top ML-Scored Puts")
            top_puts = df_with_scores[df_with_scores['put_moneyness'] == 'OTM'].sort_values('ml_score', ascending=False).head(10)
            st.dataframe(
                top_puts[['strike', 'put_ltp', 'put_iv', 'put_oi_change', 'ml_score']]
                .rename(columns={
                    'strike': 'Strike',
                    'put_ltp': 'Price',
                    'put_iv': 'IV',
                    'put_oi_change': 'OI Change',
                    'ml_score': 'ML Score'
                })
                .style.format({
                    'Price': '{:.2f}',
                    'IV': '{:.1f}%',
                    'OI Change': '{:,}',
                    'ML Score': '{:.2f}'
                }),
                use_container_width=True
            )

if __name__ == "__main__":
    main()
