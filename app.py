import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Streamlit page config and theme
st.set_page_config(
    page_title="EV Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for app UI, sidebar, and metric cards ===
st.markdown("""
<style>
body {
    background-color: #f3f7fa;
    color: #20232a;
}
.stApp {
    background: linear-gradient(115deg, #b5d0f9 20%, #f6e6fa 80%);
}
.metric-card {
    min-height: 135px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 4px 18px 0 rgba(40, 60, 100, 0.08);
    padding: 25px 0px 12px 0px;
    margin: 0 4px;
    color: #161616; /* Ensures black text by default */
}
.stSelectbox, .stMultiSelect {
    font-size: 18px;
}
h1, h2 {
    color: #183153 !important;
}
.metric {
    font-size: 32px; font-weight: 650; color: #183153;
}
.stSuccess {
    background-color: #e7f8ec !important;
    color: #058c42 !important;
    font-weight: 600;
}
.signature {
    text-align: right;
    margin-top: 2em;
    font-size: 1rem;
    color: #21324c;
    font-style: italic;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar - "App Controls" with white text on blue background
with st.sidebar:
    st.markdown(
        "<h2 style='color: #fff; background: #1ea7fd; padding: 13px 10px; border-radius: 10px; text-align: center;'>⚙️ App Controls</h2>",
        unsafe_allow_html=True
    )
    st.image("ev-car-factory.jpg")
    st.write("Select a county to see its EV forecast. Use advanced multi-county comparison below.")

# === Title and subtitle section (centered) ===
st.markdown(
    """
    <div style='text-align: center; margin-top: 0.5em'>
        <h1 style='font-size: 2.3rem; font-weight: 700;'>
            🔮 EV Adoption Forecaster for Washington State Counties
        </h1>
        <p style="font-size: 1.35rem; font-weight: 490; margin-top: -0.9em;">
            Welcome to your interactive Electric Vehicle (EV) Adoption Forecast tool
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# === Load model and data ===
model = joblib.load(r'forecasting_ev_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique())

# Sidebar selector
county = st.sidebar.selectbox("Select County:", county_list)
forecast_horizon = 36  # months

if county not in county_list:
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date").copy()
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting logic ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()
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

# Combine historical and forecast for cumulative plot
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

# === Dashboard: metrics summary (aligned cards, now black text) ===
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""<div class='metric-card'>
                <div style='font-size:19px; color:#161616;'>Historical EVs</div>
                <div style='font-size:32px; color:#161616;'>{int(historical_cum['Cumulative EV'].iloc[-1]):,}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class='metric-card'>
                <div style='font-size:19px; color:#161616;'>Forecasted Total in 3 Years</div>
                <div style='font-size:32px; color:#161616;'>{int(forecast_df['Cumulative EV'].iloc[-1]):,}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with c3:
    pct_growth = ((forecast_df['Cumulative EV'].iloc[-1] - historical_cum['Cumulative EV'].iloc[-1]) / historical_cum['Cumulative EV'].iloc[-1]) * 100 if historical_cum['Cumulative EV'].iloc[-1] else 0
    grow_icon = "📈" if pct_growth >= 0 else "📉"
    st.markdown(
        f"""<div class='metric-card'>
                <div style='font-size:19px; color:#161616;'>Projected Growth (%)</div>
                <div style='font-size:32px; color:#161616;'>{grow_icon} {pct_growth:.1f}%</div>
            </div>""",
        unsafe_allow_html=True,
    )

# === Graph interpretation and axis info ===
st.markdown("---")
st.markdown(
    """
    <div style="background: #f8fafc; border-radius: 12px; padding: 18px 16px; margin-top: 0.5em; color: #161616; font-size: 1.12rem;">
        <b>How to read this graph:</b><br>
        <ul style="margin:0;padding-left:1.2em;">
            <li><b>X-axis (horizontal):</b> Shows the progression of months and years, starting with past data and moving forward to 3-year forecasts.</li>
            <li><b>Y-axis (vertical):</b> Shows the total cumulative number of electric vehicles (EVs) registered in the county at each date.</li>
        </ul>
        The lines on the chart display both actual EV numbers until today and predicted numbers for the coming years.<br><br>
        <i>Note: All the counties listed in this tool are from North America.</i>
    </div>
    """,
    unsafe_allow_html=True
)

# === Plot Cumulative Graph ===
st.subheader(f"📊 3-Year EV Forecast for {county} County")
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(13, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'],
            label=f"{label}", marker='o', linewidth=2.5)
ax.set_title(f"Cumulative EV Trend - {county}", fontsize=19, weight='bold', color='#183153')
ax.set_xlabel("Date", fontsize=13, color="#161616")
ax.set_ylabel("Cumulative EV Count", fontsize=13, color="#161616")
ax.grid(axis="y", alpha=0.2)
ax.legend()
plt.xticks(rotation=25, fontsize=11)
plt.yticks(fontsize=12)
st.pyplot(fig)

# === County comparison - stylish and responsive ===
st.markdown("---")
st.header("🔄 County Comparison: EV Adoption Trends")

multi_counties = st.multiselect(
    "Select up to 3 counties for comparison",
    county_list,
    default=[county],
    max_selections=3,
    key="compare_counties"
)

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
    st.subheader("🟦 Side-by-Side EV Adoption Trajectories")
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ['#1ea7fd', '#ff7851', '#42d6a4']
    for idx, (cty, group) in enumerate(comp_df.groupby('County')):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty, linewidth=2.5, color=colors[idx])
    ax.set_title("EV Adoption Trends: Past + 3-Year Forecast", fontsize=18, weight='bold', color='#183153')
    ax.set_xlabel("Date", fontsize=12, color="#161616")
    ax.set_ylabel("Cumulative EV Count", fontsize=12, color="#161616")
    ax.legend(title="County", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=25, fontsize=11)
    plt.yticks(fontsize=12)
    st.pyplot(fig)
    # Growth summary display
    st.markdown("### 🌱 Projected Growth (% over next 3 years):")
    growth_summaries = []
    for idx, cty in enumerate(multi_counties):
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        forecasted_total = cty_df['Cumulative EV'].iloc[-1]
        grow_pct = ((forecasted_total - historical_total) / historical_total) * 100 if historical_total else 0
        grow_icon = "📈" if grow_pct >= 0 else "📉"
        summary = f"<b style='color:{colors[idx]}'>{cty}:</b> {grow_icon} {grow_pct:.1f}%"
        growth_summaries.append(summary)
    st.markdown(" &nbsp; | &nbsp; ".join(growth_summaries), unsafe_allow_html=True)

st.success("Forecast complete — Happy visualizing!")

# === Signature at the end ===
st.markdown(
    "<div class='signature'>Done by Karthik B</div>",
    unsafe_allow_html=True
)
