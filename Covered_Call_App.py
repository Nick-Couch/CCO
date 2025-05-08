import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
import math
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

# Page configuration for wider layout
st.set_page_config(page_title="Covered Call Optimizer", layout="wide")

# App Title
st.title("📈 Covered Call Optimizer")

# Sidebar with Tabs
tabs = st.sidebar.tabs(["Parameters", "Scoring Weights", "Multi-Contract Strategies"])
param_tab, weight_tab, multi_tab = tabs

# --- Parameters Tab ---
with param_tab:
    st.header("🔧 Parameters")
    ticker_input = st.text_input("Ticker Symbol", "AAPL")
    purchase_price = st.number_input("Purchase Price (optional)", min_value=0.0, step=0.01, value=0.0)
    basis_price = purchase_price if purchase_price > 0 else None
    target_roi_input = st.number_input("Target ROI (%) (optional)", min_value=0.0, step=0.1, value=0.0)
    target_roi = target_roi_input / 100 if target_roi_input > 0 else 0.0
    min_dte, max_dte = st.slider("Expiration Range (Days)", 1, 180, (7, 30))
    include_itm = st.checkbox("Include In-The-Money Calls", True)
    exclude_earn = st.checkbox("Exclude options near earnings", False)
    if exclude_earn:
        earn_days = st.number_input("Days from earnings to exclude", min_value=0, max_value=30, value=3)
    else:
        earn_days = 0
    exclude_div = st.checkbox("Exclude options near dividends", False)
    if exclude_div:
        div_days = st.number_input("Days from dividend to exclude", min_value=0, max_value=30, value=3)
    else:
        div_days = 0

# --- Scoring Weights Tab ---
with weight_tab:
    st.header("⚖️ Scoring Weights")
    w_yield = st.slider("Weight: Premium Yield", 0.0, 1.0, 0.4)
    w_prob = st.slider("Weight: Expiry Probability (1 - Delta)", 0.0, 1.0, 0.4)
    w_dte = st.slider("Weight: Shorter Expiration", 0.0, 1.0, 0.2)

# --- Multi-Contract Strategies Tab ---
with multi_tab:
    st.header("📊 Multi-Contract Strategies")
    num_contracts = st.number_input("Number of Contracts", min_value=1, max_value=10, value=1)
    strategy_type = st.selectbox("Strategy Type", ["Single-Strike", "Staggered Expirations", "Strike Ladder"])

# Fetch ticker info
ticker = yf.Ticker(ticker_input)
today = date.today()
try:
    info = ticker.info
    company_name = info.get('shortName') or info.get('longName') or ticker_input
except:
    company_name = ticker_input
stock_price = ticker.history(period="1d")["Close"].iloc[-1]

# Display company and price
col1, col2 = st.columns(2)
col1.metric("Company", company_name)
col2.metric("Current Price", f"${stock_price:.2f}")

# Determine basis for yield
total_basis = basis_price if basis_price else stock_price

# Fetch event dates
earnings_dates = []
if exclude_earn:
    try:
        edf = ticker.get_earnings_dates(limit=8)
        earnings_dates = [pd.to_datetime(d).date() for d in edf['Earnings Date']]
    except:
        earnings_dates = []
dividend_dates = []
if exclude_div:
    try:
        divs = ticker.dividends
        dividend_dates = [d.date() for d in divs.index if d.date() >= today]
    except:
        dividend_dates = []

# Black-Scholes delta function
def bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

# Collect & filter option data
r_rate = 0.03
results = []
for exp in ticker.options:
    try:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue
        if exclude_earn and any(abs((exp_date - d).days) <= earn_days for d in earnings_dates):
            continue
        if exclude_div and any(abs((exp_date - d).days) <= div_days for d in dividend_dates):
            continue
        calls = ticker.option_chain(exp).calls
        for _, row in calls.iterrows():
            K = row['strike']
            if not include_itm and K <= stock_price:
                continue
            sigma = row.get('impliedVolatility', 0.0)
            T = max(dte/365, 0)
            delta = bs_call_delta(stock_price, K, T, r_rate, sigma)
            prob_w = 1 - delta
            premium = row.get('lastPrice', 0.0)
            ann_y = (premium / total_basis) * (365 / dte) if dte > 0 else 0.0
            descr = f"{ticker_input} C{K} expiring in {dte}d"
            results.append({
                'Expiry': exp, 'Strike': K, 'DTE': dte,
                'Prob_Worthless': prob_w, 'Ann_Yield': ann_y,
                'Premium': premium, 'Description': descr,
                'ITM': 'ITM' if K <= stock_price else 'OTM'
            })
    except:
        continue

# Process and score results
if results:
    df = pd.DataFrame(results)
    # Filter by strike-based ROI\    
    if target_roi_input > 0:
        threshold = total_basis * (1 + target_roi)
        df = df[df['Strike'] >= threshold]
    # Normalize metrics
    df['Norm_Y'] = (df['Ann_Yield'] - df['Ann_Yield'].min()) / (df['Ann_Yield'].max() - df['Ann_Yield'].min())
    df['Norm_P'] = (df['Prob_Worthless'] - df['Prob_Worthless'].min()) / (df['Prob_Worthless'].max() - df['Prob_Worthless'].min())
    df['Norm_D'] = 1 - ((df['DTE'] - df['DTE'].min()) / (df['DTE'].max() - df['DTE'].min()))
    weights = [w_yield, w_prob, w_dte]
    if sum(weights) > 0:
        weights = [w / sum(weights) for w in weights]
    else:
        weights = [1/3, 1/3, 1/3]
    df['Score'] = (weights[0] * df['Norm_Y'] + weights[1] * df['Norm_P'] + weights[2] * df['Norm_D'])

    # Single best option display
    best = df.loc[df['Score'].idxmax()]
    st.markdown("### 🏆 Best Covered Call Recommendation")
    st.markdown(f"**{best['Description']}**")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Premium (per contract)", f"${best['Premium']*100:.2f}")
    p2.metric("Days to Expiry", f"{best['DTE']}")
    p3.metric("Annual Yield", f"{best['Ann_Yield']:.2%}")
    p4.metric("Prob. Expire Worthless", f"{best['Prob_Worthless']:.1%}")

    # Multi-contract recommendation table
    if num_contracts > 1:
        if strategy_type == "Staggered Expirations":
            top_per_exp = df.sort_values('Score', ascending=False).drop_duplicates('Expiry')
            group = top_per_exp.head(int(num_contracts))
        elif strategy_type == "Strike Ladder":
            best_expiry = df.groupby('Expiry')['Score'].mean().idxmax()
            group = df[df['Expiry'] == best_expiry].nlargest(int(num_contracts), 'Score')
        else:
            group = df.nlargest(1, 'Score')
        cols = ['Description','Strike','Premium','DTE','Prob_Worthless','Ann_Yield','Score']
        tg = group[cols].copy()
        tg['Premium'] = tg['Premium']*100
        avg = {'Description':'Average','Strike':tg['Strike'].mean(),'Premium':tg['Premium'].sum(),
               'DTE':tg['DTE'].mean(),'Prob_Worthless':tg['Prob_Worthless'].mean(),
               'Ann_Yield':tg['Ann_Yield'].mean(),'Score':tg['Score'].mean(),}
        tdf = pd.concat([tg,pd.DataFrame([avg])],ignore_index=True)
        tdf['Premium']=tdf['Premium'].map(lambda x:f"${x:.2f}")
        tdf['Prob_Worthless']=tdf['Prob_Worthless'].map(lambda x:f"{x:.1%}")
        tdf['Ann_Yield']=tdf['Ann_Yield'].map(lambda x:f"{x:.2%}")
        tdf['Score']=tdf['Score'].map(lambda x:f"{x:.2f}")
        tdf['DTE']=tdf['DTE'].map(lambda x:f"{x:.0f}")
        tdf['Strike']=tdf['Strike'].map(lambda x:f"{x:.2f}")
        st.markdown("### 📑 Multi-Contract Recommendation")
        st.table(tdf)

        # Bubble chart visible by default
    df['Prob_Percent'] = df['Prob_Worthless'] * 100
    df['Premium_100'] = df['Premium'] * 100
    bubble_fig = px.scatter(
        df,
        x='Ann_Yield',
        y='Prob_Percent',
        size='Norm_D',
        color='ITM',
        color_discrete_map={'ITM': 'green', 'OTM': 'red'},
        custom_data=['Description', 'Premium_100'],
        labels={
            'Ann_Yield': 'Annualized Yield',
            'Prob_Percent': 'Prob. Expire Worthless (%)',
            'Norm_D': 'Time Sensitivity'
        },
        title='Option Efficiency: Annual Yield vs Risk'
    )
    bubble_fig.update_traces(
        hovertemplate="%{customdata[0]}<br>Premium: $%{customdata[1]:.2f}<br>Ann Yield: %{x:.2%}<br>Prob Worthless: %{y:.1f}%"
    )
    # Highlight best option
    bubble_fig.add_trace(go.Scatter(
        x=[best['Ann_Yield']],
        y=[best['Prob_Worthless'] * 100],
        mode='markers',
        marker=dict(color='gold', size=15, line=dict(color='black', width=2)),
        name='Best Option',
        customdata=[[best['Description'], best['Premium'] * 100]],
        hovertemplate="%{customdata[0]}<br>Premium: $%{customdata[1]:.2f}<br>Ann Yield: %{x:.2%}<br>Prob Worthless: %{y:.1f}%<extra></extra>"
    ))
    st.plotly_chart(bubble_fig, use_container_width=True)

        # Collapsible option chain data table
    with st.expander("Show Option Chain Data", expanded=False):
        df2 = df.copy()
        df2['Premium'] = df2['Premium'] * 100
        st.dataframe(df2.sort_values('Score', ascending=False).reset_index(drop=True))
else:
    st.warning("No options found or none meet criteria.")
