import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.express as px

# -------------------------------
# SNOWFLAKE SESSION
# -------------------------------

# -------------------------------
# CUSTOM COLORS
# -------------------------------
custom_colors = [
    "#2F76B9","#3B9790","#F5BA2E","#6A4C93",
    "#F77F00","#B4BBBE","#e6657b","#026df5","#5aede2"
]

st.set_page_config(
    page_title="Lead Prediction Calculator",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(
        "leads_prediction.csv",
        encoding='utf-8-sig'
    )

    df.columns = df.columns.str.strip().str.upper()

    df.rename(columns={
        'MONTH_YEAR':'month_year',
        'LEADS':'Leads',
        'HIRED':'Hired'
    }, inplace=True)

    df['month_year'] = pd.to_datetime(df['month_year'])

    df = df.groupby(
        ['month_year','CAMPAIGN_SITE','BROADSOURCE'],
        as_index=False
    ).agg({
        'Leads':'sum',
        'Hired':'sum'
    })

    df['conversion_rate'] = df['Hired'] / df['Leads']

    df['conversion_rate'] = df['conversion_rate'].replace(
        [np.inf,-np.inf],
        0
    ).fillna(0)

    return df

df = load_data()

# -------------------------------
# TIME SETUP
# -------------------------------
current_month = df['month_year'].max()

prediction_month = current_month + pd.DateOffset(months=1)

# -------------------------------
# TRAIN DATA
# -------------------------------
train_df = df[df['month_year'] <= current_month]

# -------------------------------
# ARIMA MODEL
# -------------------------------
@st.cache_data
def run_arima(train_df):

    predictions = []

    for (site,source),group in train_df.groupby(
        ['CAMPAIGN_SITE','BROADSOURCE']
    ):

        ts = group.sort_values(
            'month_year'
        ).set_index(
            'month_year'
        )['Leads']

        if len(ts) < 3:
            continue

        try:

            model = ARIMA(
                ts,
                order=(1,1,1)
            )

            model_fit = model.fit()

            forecast = model_fit.forecast(
                steps=1
            )[0]

            predictions.append({
                'CAMPAIGN_SITE':site,
                'BROADSOURCE':source,
                'Predicted_Leads':max(float(forecast),0)
            })

        except:
            continue

    # -------------------------------
    # FIX FOR EMPTY DATAFRAME
    # -------------------------------
    if len(predictions) == 0:

        return pd.DataFrame(columns=[
            'CAMPAIGN_SITE',
            'BROADSOURCE',
            'Predicted_Leads'
        ])

    return pd.DataFrame(predictions)

pred_df = run_arima(train_df)

# -------------------------------
# HISTORICAL METRICS
# -------------------------------
hist = df.groupby(
    ['CAMPAIGN_SITE','BROADSOURCE']
).agg({
    'Leads':'sum',
    'Hired':'sum'
}).reset_index()

hist['share_hired'] = hist['Hired'] / hist.groupby(
    'CAMPAIGN_SITE'
)['Hired'].transform('sum')

hist['conversion_rate'] = hist['Hired'] / hist['Leads']

hist = hist.replace(
    [np.inf,-np.inf],
    0
).fillna(0)

# -------------------------------
# FINAL LEADS FUNCTION
# -------------------------------
def compute_final_leads(base,df,site=None):

    results = []

    for _,row in base.iterrows():

        source = row['BROADSOURCE']

        required = float(
            row.get('required_leads',0)
        )

        predicted = float(
            row.get('Predicted_Leads',0)
        )

        if np.isnan(required) or np.isinf(required):
            required = 0

        if np.isnan(predicted) or np.isinf(predicted):
            predicted = 0

        # -------------------------------
        # FINAL LOGIC
        # -------------------------------
        final = max(required,predicted)

        if site:

            max_leads = df[
                (df['CAMPAIGN_SITE']==site) &
                (df['BROADSOURCE']==source)
            ]['Leads'].max()

        else:

            max_leads = df[
                df['BROADSOURCE']==source
            ]['Leads'].max()

        if pd.isna(max_leads):
            limit = final
        else:
            limit = 1.5 * float(max_leads)

        capped = min(final,limit)

        excess = final - capped

        results.append({
            'BROADSOURCE':source,
            'Lead Count Required':capped,
            'excess':excess
        })

    final_df = pd.DataFrame(results)

    if 'Social Media' in final_df['BROADSOURCE'].values:

        excess_total = final_df['excess'].sum()

        final_df.loc[
            final_df['BROADSOURCE']=='Social Media',
            'Lead Count Required'
        ] += excess_total

    return final_df[
        ['BROADSOURCE','Lead Count Required']
    ]

# -------------------------------
# ROLLING ACCURACY
# -------------------------------
rolling_results = []

for i in range(3,0,-1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[
        df['month_year'] < test_month
    ]

    test_temp = df[
        df['month_year'] == test_month
    ]

    pred_temp = run_arima(train_temp)

    base = test_temp.copy()

    base = base.merge(
        pred_temp,
        on=['CAMPAIGN_SITE','BROADSOURCE'],
        how='left'
    )

    base['Predicted_Leads'] = base[
        'Predicted_Leads'
    ].fillna(0)

    base['required_leads'] = (
        base['Hired'] /
        base['conversion_rate']
    )

    base['required_leads'] = base[
        'required_leads'
    ].replace(
        [np.inf,-np.inf],
        0
    ).fillna(0)

    base['final_leads'] = base[
        ['required_leads','Predicted_Leads']
    ].max(axis=1)

    actual_total = base['Leads'].sum()

    predicted_total = base[
        'final_leads'
    ].sum()

    rmse = abs(
        actual_total - predicted_total
    )

    mape = (
        rmse / actual_total
        if actual_total != 0 else 0
    )

    rolling_results.append({
        'Month':test_month.strftime('%Y-%m'),
        'Actual Leads':actual_total,
        'Predicted Leads (Final)':predicted_total,
        'RMSE':rmse,
        'MAPE (%)':round(mape*100,2)
    })

rolling_accuracy_df = pd.DataFrame(
    rolling_results
)

# -------------------------------
# SITE LEVEL ACCURACY
# -------------------------------
site_level_results = []

for i in range(3,0,-1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[
        df['month_year'] < test_month
    ]

    test_temp = df[
        df['month_year'] == test_month
    ]

    pred_temp = run_arima(train_temp)

    base = test_temp.copy()

    base = base.merge(
        pred_temp,
        on=['CAMPAIGN_SITE','BROADSOURCE'],
        how='left'
    )

    base['Predicted_Leads'] = base[
        'Predicted_Leads'
    ].fillna(0)

    base['required_leads'] = (
        base['Hired'] /
        base['conversion_rate']
    )

    base['required_leads'] = base[
        'required_leads'
    ].replace(
        [np.inf,-np.inf],
        0
    ).fillna(0)

    base['final_leads'] = base[
        ['required_leads','Predicted_Leads']
    ].max(axis=1)

    for site_name,grp in base.groupby(
        'CAMPAIGN_SITE'
    ):

        actual_total = grp['Leads'].sum()

        predicted_total = grp[
            'final_leads'
        ].sum()

        rmse = abs(
            actual_total - predicted_total
        )

        mape = (
            rmse / actual_total
            if actual_total != 0 else 0
        )

        site_level_results.append({
            'Month':test_month.strftime('%Y-%m'),
            'CAMPAIGN_SITE':site_name,
            'Actual Leads':actual_total,
            'Predicted Leads (Final)':predicted_total,
            'RMSE':rmse,
            'MAPE (%)':round(mape*100,2)
        })

site_level_accuracy_df = pd.DataFrame(
    site_level_results
)

# -------------------------------
# FORMAT DISPLAY
# -------------------------------
def format_numbers(df):

    cols = [
        'Actual Leads',
        'Predicted Leads (Final)',
        'RMSE'
    ]

    for col in cols:

        if col in df.columns:

            df[col] = df[col].apply(
                lambda x: f"{int(round(x)):,}"
            )

    return df

rolling_accuracy_display = format_numbers(
    rolling_accuracy_df.copy()
)

site_level_accuracy_display = format_numbers(
    site_level_accuracy_df.copy()
)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title(
    "📊 Lead Prediction Calculator (Final ML Output)"
)

st.info(
    f"📅 Prediction Month: "
    f"{prediction_month.strftime('%Y-%m')}"
)

st.sidebar.header(
    "📉 Accuracy (Final Output Based)"
)

st.sidebar.dataframe(
    rolling_accuracy_display
)

st.sidebar.subheader(
    "📍 Site-Level Accuracy"
)

st.sidebar.dataframe(
    site_level_accuracy_display
)

site_options = ["All Sites"] + sorted(
    df['CAMPAIGN_SITE'].unique()
)

site = st.selectbox(
    "Select Campaign Site",
    site_options
)

target_hired = st.number_input(
    "Enter Target HIRED",
    min_value=0,
    step=1
)

# -------------------------------
# FINAL PREDICTION
# -------------------------------
if st.button("Predict"):

    if site == "All Sites":

        base = df.groupby(
            'BROADSOURCE'
        ).agg({
            'Leads':'sum',
            'Hired':'sum'
        }).reset_index()

    else:

        base = df[
            df['CAMPAIGN_SITE']==site
        ].groupby(
            'BROADSOURCE'
        ).agg({
            'Leads':'sum',
            'Hired':'sum'
        }).reset_index()

    base['share_hired'] = (
        base['Hired'] /
        base['Hired'].sum()
    )

    base['conversion_rate'] = (
        base['Hired'] /
        base['Leads']
    )

    base['target_hired'] = (
        base['share_hired'] *
        target_hired
    )

    base['required_leads'] = (
        base['target_hired'] /
        base['conversion_rate']
    )

    base['required_leads'] = base[
        'required_leads'
    ].replace(
        [np.inf,-np.inf],
        0
    ).fillna(0)

    if site == "All Sites":

        arima_agg = pred_df.groupby(
            'BROADSOURCE'
        )['Predicted_Leads'].sum().reset_index()

        base = base.merge(
            arima_agg,
            on='BROADSOURCE',
            how='left'
        )

    else:

        arima_site = pred_df[
            pred_df['CAMPAIGN_SITE']==site
        ]

        base = base.merge(
            arima_site[
                ['BROADSOURCE','Predicted_Leads']
            ],
            on='BROADSOURCE',
            how='left'
        )

    base['Predicted_Leads'] = base[
        'Predicted_Leads'
    ].fillna(0)

    output = compute_final_leads(
        base,
        df,
        site=None if site=="All Sites" else site
    )

    output['CAMPAIGN_SITE'] = site

    output['Lead Count Required'] = output[
        'Lead Count Required'
    ].round().astype(int)

    final_output = output[
        ['CAMPAIGN_SITE',
         'BROADSOURCE',
         'Lead Count Required']
    ]

    st.subheader("📈 Final Lead Plan")

    st.dataframe(final_output)

    fig = px.bar(
        final_output,
        x='BROADSOURCE',
        y='Lead Count Required',
        color='BROADSOURCE',
        color_discrete_sequence=custom_colors
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig,
        use_container_width=True
    )
