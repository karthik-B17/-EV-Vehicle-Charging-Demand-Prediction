import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# === Custom Styling for High Contrast and Better Readability ===
st.set_page_config(page_title="EV Forecast", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        color: #111 !important;
        background: #fff !important;
    }
    .stApp {
        background: #fff !important;
    }
    /* Dropdown box styling */
    .stSelectbox > div[data-baseweb="select"] {
        background-color: #111 !important;
        border-radius: 6px !important;
    }
    .stSelectbox span {
        color: #fff !important;
        font-weight: 500 !important;
    }
    .stSelectbox div[role="option"] {
        background-color: #111 !important;
        color: #fff !important;
    }
    /* Main content text color */
    .stMarkdown, .stText, .stHeader, .stSubheader {
        color: #111 !important;
    }
    /* Section headings */
    h1, h2, h3, h4, h5, h6 {
        color: #111 !important;
        font-weight: bold !important;
    }
    /* Remove blue highlight */
    .st-bd, .st-dw, .st-cs, .st-dp, .st-dt { color: #111 !important; }
    </style>
""", unsafe_allow_html=True)

# == Main Heading ==
st.markdown(
    "<h1 style='text-align:center; color:#111; font-size:2.4em; margin-top:20px;'>🔮 EV Adoption Forecaster: Washington State Counties</h1>",
    unsafe_allow_html=True
)

# == Subtitle/Instructions ==
st.markdown(
    "<div style='color:#222; padding:12px; border-radius:8px; background-color:#f0f0f0; margin-bottom:14px; margin-top:10px; font-size:1.17em; font-weight:500;'>"
    "Select a county from the dropdown below to view projected EV adoption trends for the next 3 years. For comparison, you can select up to three counties in the panel below."
    "</div>",
    unsafe_allow_html=True
)

# == Image (adjust path as needed) ==
st.image(r"ev-car-factory.jpg", use_column_width=True)

# === Load Model/Data ===
model = joblib.load(r'forecasting_ev_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique().tolist())

# === Custom Label for Dropdown: Black and Bold ===
st.markdown("<span style='color:#111;font-size:1.13em;font-weight:700;'>Select a County</span>", unsafe_allow_html=True)
county = st.selectbox("", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting Block ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()
forecast_horizon = 36
future_rows = []

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

# === Subheader for Plot with Black Text ===
st.markdown(f"<h3 style='color:#111;font-weight:800;'>Cumulative EV Forecast for {county} County</h3>", unsafe_allow_html=True)

# === Plot Cumulative Graph (Black axis text, white bg) ===
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(
        data['Date'], data['Cumulative EV'],
        label=label, marker='o', linewidth=3,
        markersize=7, alpha=0.92,
        color='#2874A6' if label == 'Forecast' else '#111'
    )
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=16, color='#111', weight='bold')
ax.set_xlabel("Date", color='#111', fontsize=13)
ax.set_ylabel("Cumulative EV Count", color='#111', fontsize=13)
ax.grid(True, which='both', axis='both', alpha=0.15)
ax.set_facecolor("#fff")
fig.patch.set_facecolor('#fff')
ax.legend()
ax.tick_params(colors='#111')
st.pyplot(fig)

historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

# === Black Highlight Box for Growth Info ===
if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.markdown(
        f"""
        <div style='
            background-color: #000;
            border-radius: 8px;
            padding: 18px 18px 18px 22px;
            margin-top: 10px;
            margin-bottom: 8px;
            font-size: 1.22em;
            color: #fff;
            font-weight: bold;
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
    st.markdown("<span style='color:#111;font-size:1.08em;font-weight:700;'>Select counties to compare</span>", unsafe_allow_html=True)
    multi_counties = st.multiselect("", county_list, max_selections=3)
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
        st.markdown("<h4 style='color:#111;font-weight:800;margin-top:18px;'>Comparison of Cumulative EV Adoption Trends</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(13, 6))
        for cty, group in comp_df.groupby('County'):
            ax.plot(group['Date'], group['Cumulative EV'], marker='o', linewidth=3, markersize=7, label=cty)
        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=15, color='#111', weight='bold')
        ax.set_xlabel("Date", color='#111', fontsize=13)
        ax.set_ylabel("Cumulative EV Count", color='#111', fontsize=13)
        ax.grid(True, alpha=0.12)
        ax.set_facecolor("#fff")
        fig.patch.set_facecolor('#fff')
        ax.tick_params(colors='#111')
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
                background-color: #000;
                border-radius: 8px;
                padding: 14px 18px 14px 22px;
                margin-top: 10px;
                margin-bottom: 8px;
                font-size: 1.10em;
                color: #fff;
                font-weight: bold;
            '>
                Forecasted EV adoption growth over next 3 years — {growth_sentence}
            </div>
            """,
            unsafe_allow_html=True
        )

# === Footer: Designed by Karthik ===
st.markdown(
    "<hr style='border:none; border-top:1.5px solid #d4d5e6;'>"
    "<div style='text-align:center; color:#888; font-size:1em; margin-top:14px;'>"
    "Designed by Karthik"
    "</div>",
    unsafe_allow_html=True
)
