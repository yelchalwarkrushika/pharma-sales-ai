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

# Page config
st.set_page_config(
    page_title="Pharma Sales AI",
    page_icon="💊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f5ff; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0533 0%, #3b0764 50%, #6d28d9 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.8) !important;
        font-size: 15px; padding: 8px 0;
    }

    .main-header {
        font-size: 2.5rem; font-weight: 800;
        color: #5B2D8E; text-align: center;
        margin-bottom: 5px; letter-spacing: -1px;
    }
    .sub-header {
        text-align: center; color: #888;
        margin-bottom: 30px; font-size: 1rem;
    }

    [data-testid="stMetric"] {
        background: white; border-radius: 16px;
        padding: 20px; box-shadow: 0 4px 15px rgba(91,45,142,0.1);
        border-left: 4px solid #7c3aed;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #5B2D8E !important; font-weight: 700 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: white !important; border: none;
        border-radius: 12px; padding: 12px 30px;
        font-weight: 600; font-size: 15px;
        transition: 0.3s; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124,58,237,0.4);
    }

    .stTextInput > div > div > input {
        border-radius: 12px; border: 2px solid #e5e7eb;
        padding: 12px 16px; font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.1);
    }

    h2, h3 { color: #1a1a1a; font-weight: 700; }

    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 15px rgba(0,0,0,0.06);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">💊 Pharma Sales AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML Forecasting + AI Recommendations + Drug Segmentation</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "ML Model", "Drug Segments", "AI Recommendations", "Ask AI"])
st.sidebar.markdown("---")
api_key = "your-groq-key-here"

# Load data
@st.cache_data
def load_data():
df = pd.read_csv("archive (3)/salesdaily.csv")
    df['datum'] = pd.to_datetime(df['datum'])
    drug_cols = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
    df['total_sales'] = df[drug_cols].sum(axis=1)
    df['weekday_num'] = df['datum'].dt.dayofweek
    return df, drug_cols

df, drug_cols = load_data()

# Train model
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

# DASHBOARD
if page == "Dashboard":
    st.subheader("Sales Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Daily Sales", f"{df['total_sales'].mean():.1f}")
    col3.metric("Peak Sales", f"{df['total_sales'].max():.1f}")
    col4.metric("Date Range", f"{df['datum'].dt.year.min()} - {df['datum'].dt.year.max()}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Sales Trend")
        monthly = df.groupby(['Year','Month'])['total_sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#faf5ff')
        ax.plot(range(len(monthly)), monthly['total_sales'], color='#7c3aed', linewidth=2.5)
        ax.fill_between(range(len(monthly)), monthly['total_sales'], alpha=0.1, color='#7c3aed')
        ax.set_xlabel("Month"); ax.set_ylabel("Total Sales")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.subheader("Sales by Drug Category")
        drug_totals = df[drug_cols].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#faf5ff')
        bars = ax.bar(drug_totals.index, drug_totals.values,
                     color=['#7c3aed','#9b4dca','#b06de0','#c58df5','#a855f7','#ec4899','#f472b6','#f9a8d4'])
        ax.set_xlabel("Drug"); ax.set_ylabel("Total Units")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sales by Weekday")
        weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        weekday_sales = df.groupby('Weekday Name')['total_sales'].mean().reindex(weekday_order)
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#faf5ff')
        ax.bar(weekday_sales.index, weekday_sales.values, color='#2D8E6B')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Sales by Month")
        monthly_avg = df.groupby('Month')['total_sales'].mean()
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#faf5ff')
        ax.bar(monthly_avg.index, monthly_avg.values, color='#ec4899')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig)

# ML MODEL
elif page == "ML Model":
    st.subheader("Random Forest Sales Predictor")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2*100:.1f}%")
    col2.metric("Mean Absolute Error", f"{mae:.2f}")
    st.markdown("---")

    st.subheader("Feature Importance — What Drives Sales?")
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#faf5ff')
    ax.bar(importances.index, importances.values,
           color=['#7c3aed' if i == 0 else '#a855f7' if i < 3 else '#c084fc' for i in range(len(importances))])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("30-Day Sales Forecast")
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
    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#faf5ff')
    ax.plot(future_dates, preds, color='#7c3aed', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(future_dates, preds*0.9, preds*1.1, alpha=0.15, color='#7c3aed', label='Confidence Range')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel("Date"); ax.set_ylabel("Predicted Sales")
    ax.legend(); plt.xticks(rotation=45)
    st.pyplot(fig)

# DRUG SEGMENTS
elif page == "Drug Segments":
    st.subheader("Drug Segmentation — KMeans Clustering")
    drug_profiles = df[drug_cols].describe().T[['mean','std','max']]
    scaler = StandardScaler()
    drug_scaled = scaler.fit_transform(drug_profiles)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    drug_profiles['cluster'] = kmeans.fit_predict(drug_scaled)
    cluster_names = {0:'High Demand', 1:'Medium Demand', 2:'Low Demand'}
    drug_profiles['segment'] = drug_profiles['cluster'].map(cluster_names)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(drug_profiles[['mean','std','max','segment']].round(2), use_container_width=True)
    with col2:
        colors_map = {'High Demand':'#2D8E6B','Medium Demand':'#7c3aed','Low Demand':'#ec4899'}
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#faf5ff')
        for segment, group in drug_profiles.groupby('segment'):
            ax.scatter(group['mean'], group['std'], label=segment,
                      s=300, color=colors_map.get(segment,'gray'), alpha=0.8, edgecolors='white', linewidth=2)
            for idx, row in group.iterrows():
                ax.annotate(idx, (row['mean'], row['std']),
                           textcoords="offset points", xytext=(8,8), fontsize=11, fontweight='bold')
        ax.set_xlabel("Average Daily Sales"); ax.set_ylabel("Sales Variability")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend()
        st.pyplot(fig)

# AI RECOMMENDATIONS
elif page == "AI Recommendations":
    st.subheader("AI Generated Business Recommendations")
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar!")
    else:
        if st.button("Generate AI Report"):
            with st.spinner("Analyzing data..."):
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
                        {"role": "system", "content": "You are a senior pharma business analyst. Give 5 specific actionable recommendations. Be professional and specific with numbers. No AI-sounding phrases."},
                        {"role": "user", "content": f"Analyze this pharma data:\n{summary}"}
                    ],
                    temperature=0.7, max_tokens=800
                )
                st.markdown(response.choices[0].message.content)

# ASK AI
elif page == "Ask AI":
    st.subheader("Ask Anything About Pharma Sales")
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar!")
    else:
        question = st.text_input("Ask a question:", placeholder="Which drug needs more marketing attention?")
        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                drug_totals = df[drug_cols].sum().sort_values(ascending=False)
                context = f"""
                Pharma sales dataset with {len(df)} records from {df['datum'].dt.year.min()} to {df['datum'].dt.year.max()}.
                Drug sales: {drug_totals.to_dict()}
                Avg daily sales: {df['total_sales'].mean():.2f}
                ML R² Score: {r2*100:.1f}%
                """
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a pharma sales analyst. Answer concisely and professionally based on data."},
                        {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {question}"}
                    ],
                    temperature=0.7, max_tokens=500
                )
                st.markdown(response.choices[0].message.content)