import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("leads_prediction.csv", encoding='utf-8-sig')

    df.columns = df.columns.str.strip().str.upper()

    df.rename(columns={
        'MONTH_YEAR': 'month_year',
        'LEADS': 'Leads',
        'HIRED': 'Hired'
    }, inplace=True)

    df['month_year'] = pd.to_datetime(df['month_year'])

    df = df.groupby(['month_year','CAMPAIGN_SITE','BROADSOURCE'], as_index=False).agg({
        'Leads':'sum',
        'Hired':'sum'
    })

    df['conversion_rate'] = df['Hired'] / df['Leads']
    df['conversion_rate'] = df['conversion_rate'].replace([np.inf, -np.inf], 0).fillna(0)

    return df


df = load_data()

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
current_month = df['month_year'].max()

train_end = current_month - pd.DateOffset(months=3)
test_month = current_month - pd.DateOffset(months=2)

train_df = df[df['month_year'] <= train_end]
test_df = df[df['month_year'] == test_month]

# -------------------------------
# ARIMA MODEL
# -------------------------------
@st.cache_data
def run_arima(train_df):
    predictions = []

    for (site, source), group in train_df.groupby(['CAMPAIGN_SITE','BROADSOURCE']):
        ts = group.sort_values('month_year').set_index('month_year')['Leads']

        if len(ts) < 3:
            continue

        try:
            model = ARIMA(ts, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]

            predictions.append({
                'CAMPAIGN_SITE': site,
                'BROADSOURCE': source,
                'Predicted_Leads': max(forecast, 0)
            })
        except:
            continue

    return pd.DataFrame(predictions)


pred_df = run_arima(train_df)

# -------------------------------
# MODEL ACCURACY
# -------------------------------
accuracy_df = pred_df.merge(
    test_df,
    on=['CAMPAIGN_SITE','BROADSOURCE'],
    how='inner'
)

if len(accuracy_df) > 0:
    rmse = np.sqrt(mean_squared_error(accuracy_df['Leads'], accuracy_df['Predicted_Leads']))
    mape = mean_absolute_percentage_error(accuracy_df['Leads'], accuracy_df['Predicted_Leads'])
else:
    rmse = 0
    mape = 0

# -------------------------------
# SITE-LEVEL ACCURACY (NEW)
# -------------------------------
site_accuracy_list = []

for site, group in accuracy_df.groupby('CAMPAIGN_SITE'):
    if len(group) == 0:
        continue

    rmse_site = np.sqrt(mean_squared_error(group['Leads'], group['Predicted_Leads']))
    mape_site = mean_absolute_percentage_error(group['Leads'], group['Predicted_Leads'])

    site_accuracy_list.append({
        'CAMPAIGN_SITE': site,
        'RMSE': round(rmse_site, 2),
        'MAPE (%)': round(mape_site * 100, 2)
    })

site_accuracy_df = pd.DataFrame(site_accuracy_list)

# Sort worst → best
if len(site_accuracy_df) > 0:
    site_accuracy_df = site_accuracy_df.sort_values(by='MAPE (%)', ascending=False)

# -------------------------------
# HISTORICAL METRICS
# -------------------------------
hist = df.groupby(['CAMPAIGN_SITE','BROADSOURCE']).agg({
    'Leads':'sum',
    'Hired':'sum'
}).reset_index()

hist['share_hired'] = hist['Hired'] / hist.groupby('CAMPAIGN_SITE')['Hired'].transform('sum')
hist['conversion_rate'] = hist['Hired'] / hist['Leads']
hist = hist.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------------------
# FUNCTIONS
# -------------------------------
def calculate_required_leads(site, target_hired):
    site_data = hist[hist['CAMPAIGN_SITE'] == site].copy()

    site_data['target_hired'] = site_data['share_hired'] * target_hired
    site_data['required_leads'] = site_data['target_hired'] / site_data['conversion_rate']
    site_data['required_leads'] = site_data['required_leads'].replace([np.inf, -np.inf], 0).fillna(0)

    return site_data


def apply_constraints(site_data, df):

    results = []
    site = site_data['CAMPAIGN_SITE'].iloc[0]

    for _, row in site_data.iterrows():
        source = row['BROADSOURCE']
        required = row['required_leads']

        max_leads = df[
            (df['CAMPAIGN_SITE'] == site) &
            (df['BROADSOURCE'] == source)
        ]['Leads'].max()

        limit = 1.5 * max_leads if pd.notnull(max_leads) else required

        capped = min(required, limit)
        excess = required - capped

        results.append({
            'BROADSOURCE': source,
            'capped_leads': capped,
            'excess': excess
        })

    final_df = pd.DataFrame(results)

    # Redistribute to Social Media
    excess_total = final_df['excess'].sum()

    if 'Social Media' in final_df['BROADSOURCE'].values:
        final_df.loc[
            final_df['BROADSOURCE'] == 'Social Media',
            'capped_leads'
        ] += excess_total

    return final_df


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📊 Lead Prediction Calculator (ARIMA)")

st.sidebar.header("📉 Model Accuracy")
st.sidebar.metric("RMSE", round(rmse, 2))
st.sidebar.metric("MAPE", f"{round(mape*100, 2)} %")

# All Sites option
site_options = ["All Sites"] + sorted(df['CAMPAIGN_SITE'].unique())
site = st.selectbox("Select Campaign Site", site_options)

target_hired = st.number_input("Enter Target HIRED", min_value=0, step=1)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    if site == "All Sites":
        base = hist.copy()
        base['target_hired'] = base['share_hired'] * target_hired
        base['required_leads'] = base['target_hired'] / base['conversion_rate']
        base['required_leads'] = base['required_leads'].replace([np.inf, -np.inf], 0).fillna(0)

        constrained = apply_constraints(base, df)
        output = base.merge(constrained, on='BROADSOURCE')

    else:
        base = calculate_required_leads(site, target_hired)
        constrained = apply_constraints(base, df)
        output = base.merge(constrained, on='BROADSOURCE')

    # Final formatting
    output['Share of HIRED'] = (output['share_hired'] * 100).round(2)
    output['L-H Conversion %'] = (output['conversion_rate'] * 100).round(2)
    output['Lead Count Required'] = output['capped_leads'].round().astype(int)

    final_output = output[[
        'CAMPAIGN_SITE',
        'BROADSOURCE',
        'Share of HIRED',
        'L-H Conversion %',
        'Lead Count Required'
    ]]

    st.subheader("📈 Predicted Next Month Output")
    st.dataframe(final_output)

    st.bar_chart(final_output.set_index('BROADSOURCE')['Lead Count Required'])

# -------------------------------
# SITE LEVEL ACCURACY DISPLAY
# -------------------------------
st.subheader("📍 Site-Level Model Accuracy")

if len(site_accuracy_df) > 0:
    st.dataframe(site_accuracy_df)
    st.bar_chart(site_accuracy_df.set_index('CAMPAIGN_SITE')['MAPE (%)'])
else:
    st.write("No accuracy data available.")
