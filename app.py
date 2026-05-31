import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from groq import Groq

st.set_page_config(page_title="Pharma Sales Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f0f2f6; }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stRadio > div { gap: 0px; }
    [data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }
    [data-testid="stSidebar"] .stRadio label {
        display: block !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        transition: 0.2s !important;
        cursor: pointer !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
    }

    .top-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c81 100%);
        padding: 28px 40px; border-radius: 16px;
        margin-bottom: 28px; display: flex;
        justify-content: space-between; align-items: center;
    }
    .header-title { font-size: 1.8rem; font-weight: 800; color: white; letter-spacing: -0.5px; margin: 0; }
    .header-sub { font-size: 0.85rem; color: #94a3b8; margin: 6px 0 0 0; }
    .header-badge {
        background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
        color: white; padding: 8px 18px; border-radius: 20px; font-size: 13px; font-weight: 600;
    }

    .section-title {
        font-size: 11px; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 1.2px;
        margin-bottom: 16px; padding-bottom: 8px;
        border-bottom: 1px solid #e2e8f0;
    }

    .chart-title { font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
    .chart-sub { font-size: 12px; color: #94a3b8; margin-bottom: 16px; }

    [data-testid="stMetric"] {
        background: white; border-radius: 14px;
        padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border-top: 3px solid #0f4c81;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important; font-size: 12px !important;
        font-weight: 600 !important; text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 800 !important; font-size: 2rem !important; }

    .stButton > button {
        background: #0f4c81; color: white !important; border: none;
        border-radius: 10px; padding: 12px 28px; font-weight: 600;
        font-size: 14px; transition: 0.2s; width: 100%;
    }
    .stButton > button:hover { background: #0d3d6b; transform: translateY(-1px); }

    .stTextInput > div > div > input {
        border-radius: 10px; border: 1.5px solid #e2e8f0;
        padding: 12px 16px; font-size: 14px; background: white;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0f4c81;
        box-shadow: 0 0 0 3px rgba(15,76,129,0.1);
    }

    .sidebar-logo {
        padding: 24px 16px 16px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 16px;
    }
    .sidebar-logo-text { font-size: 16px; font-weight: 700; color: white !important; }
    .sidebar-logo-sub { font-size: 11px; color: #475569 !important; margin-top: 3px; }

    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Hide chart toolbar */
    [data-testid="StyledFullScreenButton"] { display: none !important; }
    button[title="View fullscreen"] { display: none !important; }
    [data-testid="baseButton-headerNoPadding"] { display: none !important; }
    .modebar { display: none !important; }
    [data-testid="stElementToolbar"] { display: none !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div>
        <p class="header-title">Pharma Sales Intelligence Platform</p>
        <p class="header-sub">Real-time analytics · ML Forecasting · AI Recommendations · Drug Segmentation</p>
    </div>
    <div class="header-badge">Enterprise Analytics</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-logo">
    <p class="sidebar-logo-text">PSI Platform</p>
    <p class="sidebar-logo-sub">Pharma Sales Intelligence</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("NAVIGATION", ["Dashboard", "ML Model", "Drug Segments", "AI Recommendations", "Ask AI"])
st.sidebar.markdown("---")
api_key = st.secrets["GROQ_API_KEY"]
@st.cache_data
def load_data():
    df = pd.read_csv("archive (3)/salesdaily.csv")
    df['datum'] = pd.to_datetime(df['datum'])
    drug_cols = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
    df['total_sales'] = df[drug_cols].sum(axis=1)
    df['weekday_num'] = df['datum'].dt.dayofweek
    return df, drug_cols

df, drug_cols = load_data()

@st.cache_resource
def train_model(df, drug_cols):
    weekday_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    df_ml = df.copy()
    df_ml['weekday_num'] = df_ml['Weekday Name'].map(weekday_map)
    features = ['Year','Month','weekday_num','Hour','M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
    X = df_ml[features]
    y = df_ml['total_sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    r2 = r2_score(y_test, rf_pred)
    mae = mean_absolute_error(y_test, rf_pred)
    return rf, r2, mae, features

rf, r2, mae, features = train_model(df, drug_cols)

def make_chart(figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b', labelsize=10)
    ax.yaxis.label.set_color('#64748b')
    ax.xaxis.label.set_color('#64748b')
    return fig, ax

if page == "Dashboard":
    st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}", "2014 - 2019")
    col2.metric("Avg Daily Sales", f"{df['total_sales'].mean():.1f}", "units/day")
    col3.metric("Peak Daily Sales", f"{df['total_sales'].max():.1f}", "all time high")
    col4.metric("Drug Categories", f"{len(drug_cols)}", "tracked")

    st.markdown('<p class="section-title" style="margin-top:24px">Sales Analytics</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="chart-title">Monthly Sales Trend</p><p class="chart-sub">Total units sold per month across all drug categories</p>', unsafe_allow_html=True)
        monthly = df.groupby(['Year','Month'])['total_sales'].sum().reset_index()
        fig, ax = make_chart((8,4))
        ax.plot(range(len(monthly)), monthly['total_sales'], color='#0f4c81', linewidth=2)
        ax.fill_between(range(len(monthly)), monthly['total_sales'], alpha=0.08, color='#0f4c81')
        ax.set_xlabel("Month Index"); ax.set_ylabel("Total Sales")
        st.pyplot(fig)

    with col2:
        st.markdown('<p class="chart-title">Drug Category Performance</p><p class="chart-sub">Cumulative units sold per drug category</p>', unsafe_allow_html=True)
        drug_totals = df[drug_cols].sum().sort_values(ascending=False)
        fig, ax = make_chart((8,4))
        colors = ['#0f4c81','#1a6bb5','#2980b9','#3498db','#5dade2','#85c1e9','#aed6f1','#d6eaf8']
        ax.bar(drug_totals.index, drug_totals.values, color=colors)
        ax.set_xlabel("Drug Category"); ax.set_ylabel("Total Units")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="chart-title">Weekday Sales Pattern</p><p class="chart-sub">Average daily sales by day of week</p>', unsafe_allow_html=True)
        weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        weekday_sales = df.groupby('Weekday Name')['total_sales'].mean().reindex(weekday_order)
        fig, ax = make_chart((8,4))
        bars = ax.bar(weekday_sales.index, weekday_sales.values, color='#0f4c81')
        bars[weekday_sales.values.argmax()].set_color('#e74c3c')
        plt.xticks(rotation=45); ax.set_ylabel("Avg Sales")
        st.pyplot(fig)

    with col2:
        st.markdown('<p class="chart-title">Seasonal Sales Pattern</p><p class="chart-sub">Average sales by month — identify peak seasons</p>', unsafe_allow_html=True)
        monthly_avg = df.groupby('Month')['total_sales'].mean()
        fig, ax = make_chart((8,4))
        bars = ax.bar(monthly_avg.index, monthly_avg.values, color='#0f4c81')
        bars[monthly_avg.values.argmax()].set_color('#e74c3c')
        ax.set_xlabel("Month"); ax.set_ylabel("Avg Sales")
        st.pyplot(fig)

elif page == "ML Model":
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2*100:.1f}%", "accuracy")
    col2.metric("Mean Absolute Error", f"{mae:.2f}", "units")
    col3.metric("Algorithm", "Random Forest", "100 estimators")

    st.markdown('<p class="section-title" style="margin-top:24px">Feature Analysis</p>', unsafe_allow_html=True)
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = make_chart((12,5))
    colors = ['#0f4c81' if i == 0 else '#2980b9' if i < 3 else '#85c1e9' for i in range(len(importances))]
    ax.bar(importances.index, importances.values, color=colors)
    ax.set_title("Feature Importance — Drivers of Pharma Sales", fontsize=13, fontweight='600', color='#0f172a', pad=15)
    ax.set_ylabel("Importance Score")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown('<p class="section-title" style="margin-top:24px">30-Day Sales Forecast</p>', unsafe_allow_html=True)
    future_dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
    future_df = pd.DataFrame({
        'Year': future_dates.year, 'Month': future_dates.month,
        'weekday_num': future_dates.dayofweek,
        'Hour': [df['Hour'].mean()] * 30,
        'M01AB': [df['M01AB'].mean()] * 30, 'M01AE': [df['M01AE'].mean()] * 30,
        'N02BA': [df['N02BA'].mean()] * 30, 'N02BE': [df['N02BE'].mean()] * 30,
        'N05B': [df['N05B'].mean()] * 30, 'N05C': [df['N05C'].mean()] * 30,
        'R03': [df['R03'].mean()] * 30, 'R06': [df['R06'].mean()] * 30,
    })
    preds = rf.predict(future_df)
    fig, ax = make_chart((12,5))
    ax.plot(future_dates, preds, color='#0f4c81', linewidth=2, marker='o', markersize=4)
    ax.fill_between(future_dates, preds*0.9, preds*1.1, alpha=0.1, color='#0f4c81', label='90% Confidence Interval')
    ax.set_xlabel("Date"); ax.set_ylabel("Predicted Sales")
    ax.legend(fontsize=11); plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Drug Segments":
    st.markdown('<p class="section-title">KMeans Drug Segmentation</p>', unsafe_allow_html=True)
    drug_profiles = df[drug_cols].describe().T[['mean','std','max']]
    scaler = StandardScaler()
    drug_scaled = scaler.fit_transform(drug_profiles)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    drug_profiles['cluster'] = kmeans.fit_predict(drug_scaled)
    cluster_names = {0:'High Demand', 1:'Medium Demand', 2:'Low Demand'}
    drug_profiles['segment'] = drug_profiles['cluster'].map(cluster_names)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="chart-title">Drug Segment Classification</p>', unsafe_allow_html=True)
        st.dataframe(drug_profiles[['mean','std','max','segment']].round(2), use_container_width=True)
    with col2:
        colors_map = {'High Demand':'#e74c3c','Medium Demand':'#0f4c81','Low Demand':'#27ae60'}
        fig, ax = make_chart((8,6))
        for segment, group in drug_profiles.groupby('segment'):
            ax.scatter(group['mean'], group['std'], label=segment,
                      s=300, color=colors_map.get(segment,'gray'), alpha=0.9, edgecolors='white', linewidth=2)
            for idx, row in group.iterrows():
                ax.annotate(idx, (row['mean'], row['std']),
                           textcoords="offset points", xytext=(8,8), fontsize=11, fontweight='bold', color='#0f172a')
        ax.set_xlabel("Average Daily Sales"); ax.set_ylabel("Sales Variability")
        ax.set_title("Drug Portfolio Segmentation Matrix", fontsize=13, fontweight='600', color='#0f172a', pad=15)
        ax.legend()
        st.pyplot(fig)

elif page == "AI Recommendations":
    st.markdown('<p class="section-title">AI-Powered Business Intelligence</p>', unsafe_allow_html=True)
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar!")
    else:
        if st.button("Generate Strategic Recommendations"):
            with st.spinner("Analyzing data with AI..."):
                drug_totals = df[drug_cols].sum().sort_values(ascending=False)
                weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                weekday_sales = df.groupby('Weekday Name')['total_sales'].mean().reindex(weekday_order)
                monthly_avg = df.groupby('Month')['total_sales'].mean()
                yearly = df.groupby('Year')['total_sales'].sum()
                yoy = ((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2]) * 100
                summary = f"""
                Best drug: {drug_totals.index[0]} ({drug_totals.iloc[0]:,.0f} units)
                Worst drug: {drug_totals.index[-1]} ({drug_totals.iloc[-1]:,.0f} units)
                Best day: {weekday_sales.idxmax()}, Worst day: {weekday_sales.idxmin()}
                Peak month: {monthly_avg.idxmax()}, Slow month: {monthly_avg.idxmin()}
                ML R² Score: {r2*100:.1f}%, YoY Growth: {yoy:.1f}%
                """
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a senior pharma business analyst at a top consulting firm. Give 5 specific strategic recommendations. Use a professional consulting tone. Be specific with numbers and percentages. Format with numbered points."},
                        {"role": "user", "content": f"Analyze this pharma sales data:\n{summary}"}
                    ],
                    temperature=0.7, max_tokens=800
                )
                st.markdown(response.choices[0].message.content)

elif page == "Ask AI":
    st.markdown('<p class="section-title">Natural Language Analytics</p>', unsafe_allow_html=True)
    st.markdown("Ask any question about the pharma sales data in plain English.")
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar!")
    else:
        question = st.text_input("", placeholder="e.g. Which drug needs more marketing attention this quarter?")
        if st.button("Analyze") and question:
            with st.spinner("Processing..."):
                drug_totals = df[drug_cols].sum().sort_values(ascending=False)
                context = f"""
                Pharma sales dataset: {len(df)} records, {df['datum'].dt.year.min()} to {df['datum'].dt.year.max()}.
                Drug sales totals: {drug_totals.to_dict()}
                Average daily sales: {df['total_sales'].mean():.2f}
                ML Model R² Score: {r2*100:.1f}%
                """
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a senior pharma sales analyst. Answer concisely and professionally. Use data to support your answer."},
                        {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {question}"}
                    ],
                    temperature=0.7, max_tokens=500
                )
                st.markdown(response.choices[0].message.content)