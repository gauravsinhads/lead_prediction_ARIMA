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
# PREDICTION MONTH
# -------------------------------
current_month = df['month_year'].max()
prediction_month = current_month + pd.DateOffset(months=1)

# -------------------------------
# TRAIN DATA
# -------------------------------
train_end = current_month - pd.DateOffset(months=3)
train_df = df[df['month_year'] <= train_end]

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
# ROLLING ACCURACY (OVERALL + SITE)
# -------------------------------
rolling_results = []
site_rolling_results = []

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[df['month_year'] < test_month]
    test_temp = df[df['month_year'] == test_month]

    pred_temp = run_arima(train_temp)

    merged = pred_temp.merge(
        test_temp,
        on=['CAMPAIGN_SITE','BROADSOURCE'],
        how='inner'
    )

    if len(merged) == 0:
        continue

    # Overall
    rmse = np.sqrt(mean_squared_error(merged['Leads'], merged['Predicted_Leads']))
    mape = mean_absolute_percentage_error(merged['Leads'], merged['Predicted_Leads'])

    rolling_results.append({
        'Month': test_month.strftime('%Y-%m'),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape * 100, 2)
    })

    # Site-level
    for site, group in merged.groupby('CAMPAIGN_SITE'):

        if len(group) == 0:
            continue

        rmse_site = np.sqrt(mean_squared_error(group['Leads'], group['Predicted_Leads']))
        mape_site = mean_absolute_percentage_error(group['Leads'], group['Predicted_Leads'])

        site_rolling_results.append({
            'Month': test_month.strftime('%Y-%m'),
            'CAMPAIGN_SITE': site,
            'RMSE': round(rmse_site, 2),
            'MAPE (%)': round(mape_site * 100, 2)
        })

rolling_accuracy_df = pd.DataFrame(rolling_results)
site_rolling_accuracy_df = pd.DataFrame(site_rolling_results)

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


def apply_constraints(site_data, df, site=None):

    results = []

    for _, row in site_data.iterrows():
        source = row['BROADSOURCE']
        required = row['required_leads']

        if site:
            max_leads = df[(df['CAMPAIGN_SITE']==site)&(df['BROADSOURCE']==source)]['Leads'].max()
        else:
            max_leads = df[df['BROADSOURCE']==source]['Leads'].max()

        limit = 1.5 * max_leads if pd.notnull(max_leads) else required

        capped = min(required, limit)
        excess = required - capped

        results.append({
            'BROADSOURCE': source,
            'capped_leads': capped,
            'excess': excess
        })

    final_df = pd.DataFrame(results)

    excess_total = final_df['excess'].sum()

    if 'Social Media' in final_df['BROADSOURCE'].values:
        final_df.loc[final_df['BROADSOURCE']=='Social Media','capped_leads'] += excess_total

    return final_df

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📊 Lead Prediction Calculator (ML + ARIMA)")

st.info(f"📅 Prediction Month: {prediction_month.strftime('%Y-%m')}")

# Sidebar accuracy
st.sidebar.header("📉 Rolling Accuracy (Overall)")
st.sidebar.dataframe(rolling_accuracy_df)

st.sidebar.header("📍 Site-wise Rolling Accuracy")
st.sidebar.dataframe(site_rolling_accuracy_df)

# Inputs
site_options = ["All Sites"] + sorted(df['CAMPAIGN_SITE'].unique())
site = st.selectbox("Select Campaign Site", site_options)

target_hired = st.number_input("Enter Target HIRED", min_value=0, step=1)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    if site == "All Sites":

        base = df.groupby('BROADSOURCE').agg({
            'Leads':'sum',
            'Hired':'sum'
        }).reset_index()

        base['share_hired'] = base['Hired'] / base['Hired'].sum()
        base['conversion_rate'] = base['Hired'] / base['Leads']

        base['target_hired'] = base['share_hired'] * target_hired
        base['required_leads'] = base['target_hired'] / base['conversion_rate']
        base = base.replace([np.inf, -np.inf], 0).fillna(0)

        arima_agg = pred_df.groupby('BROADSOURCE')['Predicted_Leads'].sum().reset_index()

        base = base.merge(arima_agg, on='BROADSOURCE', how='left')
        base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

        base['final_leads'] = base[['required_leads','Predicted_Leads']].max(axis=1)

        constrained = apply_constraints(base.rename(columns={'final_leads':'required_leads'}), df, site=None)

        output = base.merge(constrained, on='BROADSOURCE')
        output['CAMPAIGN_SITE'] = "All Sites"

    else:
        base = calculate_required_leads(site, target_hired)

        arima_site = pred_df[pred_df['CAMPAIGN_SITE'] == site]

        base = base.merge(arima_site[['BROADSOURCE','Predicted_Leads']], on='BROADSOURCE', how='left')
        base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

        base['final_leads'] = base[['required_leads','Predicted_Leads']].max(axis=1)

        constrained = apply_constraints(
            base.rename(columns={'final_leads':'required_leads'}),
            df,
            site=site
        )

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
        'Predicted_Leads',
        'Lead Count Required'
    ]]

    st.subheader("📈 Predicted Output (ML + Business)")
    st.dataframe(final_output)

    st.bar_chart(final_output.set_index('BROADSOURCE')['Lead Count Required'])

# -------------------------------
# VISUALS
# -------------------------------
st.subheader("📉 Overall Rolling Accuracy Trend")
if len(rolling_accuracy_df) > 0:
    st.line_chart(rolling_accuracy_df.set_index('Month')['MAPE (%)'])

st.subheader("📍 Site-wise Avg Accuracy")
if len(site_rolling_accuracy_df) > 0:
    st.bar_chart(
        site_rolling_accuracy_df.groupby('CAMPAIGN_SITE')['MAPE (%)'].mean()
    )
