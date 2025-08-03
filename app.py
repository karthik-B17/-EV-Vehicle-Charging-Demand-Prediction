import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# === Streamlit Page Config ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Custom Styling and Fonts ===
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(110deg, #eaf6fb 0%, #f7e7fb 100%);
    }
    .stApp {
        background: linear-gradient(100deg, #f5faff 30%, #c2d3f2 100%);
    }
    .st-bd, .st-dw, .st-cs, .st-dp, .st-dt {
        color: #333333 !important;
    }
    .stButton>button {
        background: #5171A5;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 0.5em 1.3em;
        font-size: 1rem;
        font-weight: 600;
    }
    .stSelectbox>div>div>select {
        background-color: #fff;
        color: #254579;
        font-weight: 500;
    }
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-thumb {background: #b5c7e6; border-radius: 4px;}
    </style>
""", unsafe_allow_html=True)

# === Header and Subtitle ===
st.markdown(
    "<h1 style='text-align:center; color:#254579; font-size:2.6em; margin-top:20px;'>🔮 EV Adoption Forecaster: Washington State Counties</h1>",
    unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#254567; font-size:1.4em; margin-bottom:18px; margin-top:6px;'>Visualize and compare Electric Vehicle (EV) adoption forecasts for any county. Interactive, data-driven, and easy to explore!</p>",
    unsafe_allow_html=True)

st.markdown(
    "<div style='color:#4a536b; padding:16px; border-radius:8px; background-color:#f0f6fa; margin-bottom:18px;'>"
    "<b>Instructions:</b> Select a county from the dropdown below to view the projected EV adoption trend for the next 3 years. For comparison, add up to three counties in the comparison section below.</div>",
    unsafe_allow_html=True)

st.image(r"ev-car-factory.jpg", use_column_width=True)

# === Load Data and Model ===
model = joblib.load(r'forecasting_ev_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique().tolist())

# === County Dropdown ===
county = st.selectbox("🏛️ Select a County", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# === Combine Historical + Forecast for Cumulative Plot ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot Cumulative Graph ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(
        data['Date'], data['Cumulative EV'],
        label=label, marker='o', linewidth=3,
        markersize=7, alpha=0.9,
        color='#5171A5' if label == 'Forecast' else '#254579'
    )
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=18, color='#254579', weight='bold')
ax.set_xlabel("Date", color='#333333', fontsize=14)
ax.set_ylabel("Cumulative EV Count", color='#333333', fontsize=14)
ax.grid(True, which='both', axis='both', alpha=0.09)
ax.set_facecolor("#f5faff")
fig.patch.set_facecolor('#f5faff')
ax.legend()
st.pyplot(fig)

historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

# ====== THIS SECTION IS UPDATED: black box with white text ======
if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.markdown(
        f"""
        <div style='
            background-color: #000000;
            border-radius: 8px;
            padding: 18px 18px 18px 22px;
            margin-top: 10px;
            margin-bottom: 8px;
            font-size: 1.22em;
            color: #ffffff;
            box-shadow: 0 2px 4px rgba(40,40,40,0.07);
            border: 1.5px solid #181818;
        '>
            <b>{county}:</b> Projected {trend} of <b>{forecast_growth_pct:.2f}%</b> EV adoption over 3 years.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Historical EV total is zero; percentage forecast change cannot be computed.")

# === Comparison Section ===
st.markdown("---")
with st.expander("Compare up to 3 counties 🚗📊"):
    multi_counties = st.multiselect("🔎 Select up to 3 counties", county_list, max_selections=3)
    if multi_counties:
        comparison_data = []
        for cty in multi_counties:
            cty_df = df[df['County'] == cty].sort_values("Date")
            cty_code = cty_df['county_encoded'].iloc[0]
            hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
            cum_ev = list(np.cumsum(hist_ev))
            months_since = cty_df['months_since_start'].max()
            last_date = cty_df['Date'].max()
            future_rows_cty = []
            for i in range(1, forecast_horizon + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                months_since += 1
                lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                recent_cum = cum_ev[-6:]
                ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0
                new_row = {
                    'months_since_start': months_since,
                    'county_encoded': cty_code,
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_slope
                }
                pred = model.predict(pd.DataFrame([new_row]))[0]
                future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
                hist_ev.append(pred)
                if len(hist_ev) > 6:
                    hist_ev.pop(0)
                cum_ev.append(cum_ev[-1] + pred)
                if len(cum_ev) > 6:
                    cum_ev.pop(0)
            hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
            fc_df = pd.DataFrame(future_rows_cty)
            fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]
            combined_cty = pd.concat([
                hist_cum[['Date', 'Cumulative EV']],
                fc_df[['Date', 'Cumulative EV']]
            ], ignore_index=True)
            combined_cty['County'] = cty
            comparison_data.append(combined_cty)
        comp_df = pd.concat(comparison_data, ignore_index=True)
        st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, group in comp_df.groupby('County'):
            ax.plot(group['Date'], group['Cumulative EV'], marker='o', linewidth=3, markersize=7, label=cty)
        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='#254579', weight='bold')
        ax.set_xlabel("Date", color='#333333', fontsize=14)
        ax.set_ylabel("Cumulative EV Count", color='#333333', fontsize=14)
        ax.grid(True, alpha=0.09)
        ax.set_facecolor("#f5faff")
        fig.patch.set_facecolor('#f5faff')
        ax.legend(title="County")
        st.pyplot(fig)
        # Display % growth for selected counties
        growth_summaries = []
        for cty in multi_counties:
            cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
            historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
            forecasted_total = cty_df['Cumulative EV'].iloc[-1]
            if historical_total > 0:
                growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
            else:
                growth_summaries.append(f"{cty}: N/A (no historical data)")
        growth_sentence = " | ".join(growth_summaries)
        st.markdown(
            f"""
            <div style='
                background-color: #000000;
                border-radius: 8px;
                padding: 14px 18px 14px 22px;
                margin-top: 10px;
                margin-bottom: 8px;
                font-size: 1.11em;
                color: #ffffff;
                box-shadow: 0 2px 4px rgba(40,40,40,0.07);
                border: 1.5px solid #181818;
            '>
                Forecasted EV adoption growth over next 3 years — {growth_sentence}
            </div>
            """,
            unsafe_allow_html=True
        )

# === Footer ===
st.markdown(
    "<hr style='border:none; border-top:1.5px solid #d4d5e6;'>"
    "<div style='text-align:center; color:#888; font-size:0.97em;'>"
    "🚗 Powered by open data and ML | Designed by Your Name or Team"
    "</div>",
    unsafe_allow_html=True)
