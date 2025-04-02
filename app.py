import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from hmmlearn import hmm
import json

# Configure page
st.set_page_config(
    page_title="Upstox Options Chain Dashboard",
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
    }
    .trade-recommendation.sell {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
    }
    .strike-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #f5f5f5;
    }
    .regime-low {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
    }
    .regime-normal {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 5px;
    }
    .regime-high {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
    }
    .smart-money {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
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

# Fetch data from Upstox API
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

# Probability of Profit Calculator
def calculate_probability_of_profit(row, spot_price, option_type='call'):
    strike = row['strike']
    premium = row['call_ltp'] if option_type == 'call' else row['put_ltp']
    iv = row['call_iv'] if option_type == 'call' else row['put_iv']
    days_to_expiry = 7  # Example
    
    if option_type == 'call':
        break_even = strike + premium
        pop = 1 - norm.cdf(np.log(break_even/spot_price) / (iv * np.sqrt(days_to_expiry/365)))
    else:
        break_even = strike - premium
        pop = norm.cdf(np.log(break_even/spot_price) / (iv * np.sqrt(days_to_expiry/365)))
    
    return pop * 100

# Expected Move Calculation
def calculate_expected_move(spot_price, iv, days_to_expiry=7):
    return spot_price * iv * np.sqrt(days_to_expiry/365)

# Greeks-Based Alerts
def generate_greeks_alerts(row):
    alerts = []
    
    if abs(row['call_gamma']) > 0.08:
        alerts.append(("High Gamma", "Potential for gamma scalping"))
    if row['call_theta'] < -0.05:
        alerts.append(("High Theta", "Rapid time decay - caution selling"))
    if abs(row['call_vega']) > 0.15:
        alerts.append(("High Vega", "Highly sensitive to volatility changes"))
    
    return alerts

# Market Regime Detection
def detect_market_regime(historical_iv):
    try:
        # Remove NaN values and ensure we have enough data
        historical_iv = historical_iv[~np.isnan(historical_iv)]
        if len(historical_iv) < 10:
            return 1  # Return "Normal" regime if not enough data
        
        # Convert to daily changes
        returns = np.diff(np.log(historical_iv))
        
        # Remove any infinite values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 2:
            return 1  # Return "Normal" regime if not enough data
            
        # Fit HMM model
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")
        model.fit(returns.reshape(-1,1))
        
        # Predict regimes
        regimes = model.predict(returns.reshape(-1,1))
        
        return regimes[-1]
    except Exception as e:
        st.warning(f"Market regime detection failed: {str(e)}")
        return 1

# Put-Call Parity Arbitrage Detection
def detect_arbitrage_opportunities(df, spot_price, risk_free_rate=0.05):
    arbitrage_ops = []
    
    for _, row in df.iterrows():
        synthetic_put = row['call_ltp'] - spot_price + row['strike'] * np.exp(-risk_free_rate * (7/365))
        actual_put = row['put_ltp']
        
        if abs(synthetic_put - actual_put) > (0.05 * actual_put):
            direction = "Buy Put + Sell Synthetic" if synthetic_put > actual_put else "Buy Synthetic + Sell Put"
            arbitrage_ops.append({
                'strike': row['strike'],
                'premium_diff': synthetic_put - actual_put,
                'direction': direction
            })
    
    return sorted(arbitrage_ops, key=lambda x: abs(x['premium_diff']), reverse=True)

# Smart Money Flow Identification - FIXED VERSION
def detect_smart_money_flow(df):
    smart_money = []
    
    # Check if DataFrame is empty
    if df.empty:
        return smart_money
    
    # Calculate median call premium
    try:
        median_call_premium = df['call_ltp'].median()
        
        # Institutional Call Buying
        high_premium_calls = df[(df['call_oi_change'] > 0) & 
                               (df['call_ltp'] > 1.5 * median_call_premium)]
        if not high_premium_calls.empty:
            smart_money.append(("Institutional Call Buying", high_premium_calls['strike'].tolist()))
        
        # Put Writing
        put_writing = df[(df['put_oi_change'] < 0) & 
                        (df['put_ltp'] > df['put_ltp'].quantile(0.75))]
        if not put_writing.empty:
            smart_money.append(("Put Writing (Bearish)", put_writing['strike'].tolist()))
    
    except Exception as e:
        st.warning(f"Smart money detection failed: {str(e)}")
    
    return smart_money

# Machine Learning Trade Signals
def generate_ml_trade_signals(df, spot_price):
    try:
        # Feature engineering
        df['distance_from_spot'] = (df['strike'] - spot_price) / spot_price
        df['iv_skew'] = df['call_iv'] - df['put_iv']
        df['premium_ratio'] = (df['call_ask'] - df['call_bid']) / df['call_ltp'].replace(0, 0.01)  # Handle division by zero
        
        # Create labels (1 = good buy, 0 = avoid)
        conditions = [
            (df['call_oi_change'] > df['call_oi_change'].quantile(0.75)) &
            (df['premium_ratio'] < 0.1) &
            (df['call_iv'] < df['call_iv'].quantile(0.75)),
            
            (df['put_oi_change'] > df['put_oi_change'].quantile(0.75)) &
            (df['premium_ratio'] < 0.1) &
            (df['put_iv'] < df['put_iv'].quantile(0.75))
        ]
        choices = [1, 1]
        df['target'] = np.select(conditions, choices, default=0)
        
        # Train model
        features = ['distance_from_spot', 'iv_skew', 'premium_ratio', 
                   'call_oi_change', 'put_oi_change', 'call_iv', 'put_iv']
        X = df[features].fillna(0)
        y = df['target']
        
        if len(X) > 0 and len(y) > 0:
            model = GradientBoostingClassifier()
            model.fit(X, y)
            
            # Generate predictions
            df['ml_score'] = model.predict_proba(X)[:,1]
            
            return df.sort_values('ml_score', ascending=False)
    
    except Exception as e:
        st.error(f"ML signal generation failed: {str(e)}")
        return df
    
    return df

# Generate trade recommendations
def generate_trade_recommendations(df, spot_price):
    recommendations = []
    
    try:
        # Calculate metrics for all strikes
        df['call_premium_ratio'] = (df['call_ask'] - df['call_bid']) / df['call_ltp'].replace(0, 0.01)
        df['put_premium_ratio'] = (df['put_ask'] - df['put_bid']) / df['put_ltp'].replace(0, 0.01)
        df['call_risk_reward'] = (spot_price - df['strike'] + df['call_ltp']) / df['call_ltp'].replace(0, 0.01)
        df['put_risk_reward'] = (df['strike'] - spot_price + df['put_ltp']) / df['put_ltp'].replace(0, 0.01)
        
        # Find best calls to buy
        best_calls = df[(df['call_moneyness'] == 'OTM') & 
                       (df['call_premium_ratio'] < 0.1) &
                       (df['call_oi_change'] > 0)].sort_values(
            by=['call_premium_ratio', 'call_oi_change'], 
            ascending=[True, False]
        ).head(3)
        
        for _, row in best_calls.iterrows():
            pop = calculate_probability_of_profit(row, spot_price, 'call')
            expected_move = calculate_expected_move(spot_price, row['call_iv'])
            greeks_alerts = generate_greeks_alerts(row)
            
            recommendations.append({
                'type': 'BUY CALL',
                'strike': row['strike'],
                'premium': row['call_ltp'],
                'iv': row['call_iv'],
                'oi_change': row['call_oi_change'],
                'risk_reward': f"{row['call_risk_reward']:.1f}:1",
                'probability_of_profit': f"{pop:.1f}%",
                'expected_move': f"±{expected_move:.1f} points",
                'greeks_alerts': greeks_alerts,
                'reason': "Low spread, OI buildup, good risk/reward"
            })
        
        # Find best puts to buy
        best_puts = df[(df['put_moneyness'] == 'OTM') & 
                      (df['put_premium_ratio'] < 0.1) &
                      (df['put_oi_change'] > 0)].sort_values(
            by=['put_premium_ratio', 'put_oi_change'], 
            ascending=[True, False]
        ).head(3)
        
        for _, row in best_puts.iterrows():
            pop = calculate_probability_of_profit(row, spot_price, 'put')
            expected_move = calculate_expected_move(spot_price, row['put_iv'])
            greeks_alerts = generate_greeks_alerts(row)
            
            recommendations.append({
                'type': 'BUY PUT',
                'strike': row['strike'],
                'premium': row['put_ltp'],
                'iv': row['put_iv'],
                'oi_change': row['put_oi_change'],
                'risk_reward': f"{row['put_risk_reward']:.1f}:1",
                'probability_of_profit': f"{pop:.1f}%",
                'expected_move': f"±{expected_move:.1f} points",
                'greeks_alerts': greeks_alerts,
                'reason': "Low spread, OI buildup, good risk/reward"
            })
        
        # Find best calls to sell
        best_sell_calls = df[(df['call_moneyness'] == 'ITM') & 
                            (df['call_premium_ratio'] > 0.15) &
                            (df['call_oi_change'] < 0)].sort_values(
            by=['call_premium_ratio', 'call_oi_change'], 
            ascending=[False, True]
        ).head(2)
        
        for _, row in best_sell_calls.iterrows():
            pop = calculate_probability_of_profit(row, spot_price, 'call')
            greeks_alerts = generate_greeks_alerts(row)
            
            recommendations.append({
                'type': 'SELL CALL',
                'strike': row['strike'],
                'premium': row['call_ltp'],
                'iv': row['call_iv'],
                'oi_change': row['call_oi_change'],
                'risk_reward': f"{1/row['call_risk_reward']:.1f}:1",
                'probability_of_profit': f"{100-pop:.1f}%",
                'greeks_alerts': greeks_alerts,
                'reason': "High spread, OI unwinding, favorable risk"
            })
        
        # Find best puts to sell
        best_sell_puts = df[(df['put_moneyness'] == 'ITM') & 
                           (df['put_premium_ratio'] > 0.15) &
                           (df['put_oi_change'] < 0)].sort_values(
            by=['put_premium_ratio', 'put_oi_change'], 
            ascending=[False, True]
        ).head(2)
        
        for _, row in best_sell_puts.iterrows():
            pop = calculate_probability_of_profit(row, spot_price, 'put')
            greeks_alerts = generate_greeks_alerts(row)
            
            recommendations.append({
                'type': 'SELL PUT',
                'strike': row['strike'],
                'premium': row['put_ltp'],
                'iv': row['put_iv'],
                'oi_change': row['put_oi_change'],
                'risk_reward': f"{1/row['put_risk_reward']:.1f}:1",
                'probability_of_profit': f"{100-pop:.1f}%",
                'greeks_alerts': greeks_alerts,
                'reason': "High spread, OI unwinding, favorable risk"
            })
    
    except Exception as e:
        st.error(f"Trade recommendation generation failed: {str(e)}")
    
    return recommendations

# Main App
def main():
    st.markdown("<div class='header'><h1>📊 Upstox Options Chain Dashboard</h1></div>", unsafe_allow_html=True)
    
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
        
        st.markdown("---")
        st.markdown(f"**Current Nifty Spot Price: {spot_price:,.2f}**")
        
        st.markdown("---")
        st.markdown("**Analysis Settings**")
        volume_threshold = st.number_input("High Volume Threshold", value=5000000)
        oi_change_threshold = st.number_input("Significant OI Change", value=1000000)
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("This dashboard provides real-time options chain analysis using Upstox API data.")
    
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
        
        # PCR gauge
        pcr = df[df['strike'] == selected_strike]['pcr'].values[0]
        fig = px.bar(x=[pcr], range_x=[0, 2], title=f"Put-Call Ratio: {pcr:.2f}")
        fig.update_layout(
            xaxis_title="PCR",
            yaxis_visible=False,
            height=150,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.add_vline(x=0.7, line_dash="dot", line_color="green")
        fig.add_vline(x=1.3, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Market Regime Analysis
    try:
        regime = detect_market_regime(df['call_iv'].values)
        regimes = ["Low Volatility", "Normal", "High Volatility"]
        regime_class = ["regime-low", "regime-normal", "regime-high"][regime]
        
        st.markdown(f"""
            <div class='{regime_class}'>
                <h3>Market Regime: {regimes[regime]}</h3>
                <p>Implied Volatility: {df['call_iv'].mean():.1f}% | Expected Daily Move: ±{calculate_expected_move(spot_price, df['call_iv'].mean()):.1f} points</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not determine market regime: {str(e)}")
    
    # Smart Money Flow
    try:
        smart_money = detect_smart_money_flow(df)
        if smart_money:
            st.markdown("### Smart Money Flow")
            for flow in smart_money:
                st.markdown(f"""
                    <div class='smart-money'>
                        <b>{flow[0]}</b> detected at strikes: {', '.join(map(str, flow[1]))}
                    </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not detect smart money flow: {str(e)}")
    
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
    
    # Trade Recommendations
    st.markdown("### Trade Recommendations")
    recommendations = generate_trade_recommendations(df, spot_price)
    
    if recommendations:
        for rec in recommendations:
            is_sell = 'SELL' in rec['type']
            st.markdown(f"""
                <div class='trade-recommendation{' sell' if is_sell else ''}'>
                    <h4>{rec['type']} @ {rec['strike']:.0f}</h4>
                    <p>
                        Premium: {rec['premium']:.2f} | IV: {rec['iv']:.1f}%<br>
                        OI Change: {rec['oi_change']:,} | Risk/Reward: {rec['risk_reward']}<br>
                        Probability of Profit: {rec['probability_of_profit']} | Expected Move: {rec['expected_move']}<br>
                        {''.join([f"<b>Alert:</b> {alert[0]} - {alert[1]}<br>" for alert in rec.get('greeks_alerts', [])])}
                        <b>Reason:</b> {rec['reason']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No strong trade recommendations based on current market conditions")
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Strike Analysis", "Advanced Analytics", "Predictive Models"])
    
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
        st.markdown("### Advanced Analytics")
        
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
        
        # Greeks Monitoring
        st.markdown("#### Greeks Monitoring")
        greeks_cols = st.columns(3)
        with greeks_cols[0]:
            st.plotly_chart(px.line(df, x='strike', y='call_gamma', title='Call Gamma Exposure'))
        with greeks_cols[1]:
            st.plotly_chart(px.line(df, x='strike', y='call_theta', title='Call Theta Decay'))
        with greeks_cols[2]:
            st.plotly_chart(px.line(df, x='strike', y='call_vega', title='Call Vega Sensitivity'))
        
        # Arbitrage Opportunities
        st.markdown("#### Arbitrage Opportunities")
        arbitrage_ops = detect_arbitrage_opportunities(df, spot_price)
        if arbitrage_ops:
            for op in arbitrage_ops[:3]:
                st.warning(f"Strike {op['strike']}: {op['direction']} (Premium Diff: {op['premium_diff']:.2f})")
        else:
            st.info("No arbitrage opportunities detected")
    
    with tab3:
        st.markdown("### Predictive Models")
        
        # Machine Learning Trade Signals
        st.markdown("#### Machine Learning Trade Signals")
        ml_signals = generate_ml_trade_signals(df.copy(), spot_price)
        st.dataframe(
            ml_signals[['strike', 'ml_score', 'call_oi_change', 'put_oi_change', 'call_iv', 'put_iv']].head(10).style.background_gradient(subset=['ml_score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Probability Distribution
        st.markdown("#### Probability Distribution")
        strike_range = np.linspace(spot_price * 0.9, spot_price * 1.1, 20)
        prob_up = [calculate_probability_of_profit(df.iloc[(df['strike'] - s).abs().argsort()[:1].iloc[0]], spot_price, 'call') for s in strike_range]
        prob_down = [calculate_probability_of_profit(df.iloc[(df['strike'] - s).abs().argsort()[:1].iloc[0]], spot_price, 'put') for s in strike_range]
        
        fig = px.line(x=strike_range, y=[prob_up, prob_down], 
                     labels={'x': 'Strike Price', 'value': 'Probability (%)'},
                     title='Probability of Profit Across Strikes')
        fig.update_traces(line_color='green', selector={'name': '0'})
        fig.update_traces(line_color='red', selector={'name': '1'})
        fig.add_vline(x=spot_price, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
