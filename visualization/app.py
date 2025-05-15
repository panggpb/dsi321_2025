import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pyarrow.parquet as pq
import s3fs
import time
from openai import OpenAI
from langchain.prompts import PromptTemplate
from zoneinfo import ZoneInfo
from datetime import timedelta, datetime


# Set up environments of LakeFS
lakefs_endpoint = os.getenv("LAKEFS_ENDPOINT", "http://lakefs-dev:8000")
ACCESS_KEY = os.getenv("LAKEFS_ACCESS_KEY")
SECRET_KEY = os.getenv("LAKEFS_SECRET_KEY")

# Setting S3FileSystem for access LakeFS
fs = s3fs.S3FileSystem(
    key=ACCESS_KEY,
    secret=SECRET_KEY,
    client_kwargs={'endpoint_url': lakefs_endpoint}
)

@st.cache_data()
def load_data():
    lakefs_path = "s3://air-quality/main/airquality.parquet/year=2025"
    data_list = fs.glob(f"{lakefs_path}/*/*/*/*")
    df_all = pd.concat([pd.read_parquet(f"s3://{path}", engine="pyarrow", filesystem=fs) for path in data_list], ignore_index=True)

    # Change Data Type
    df_all['lat'] = pd.to_numeric(df_all['lat'], errors='coerce')
    df_all['long'] = pd.to_numeric(df_all['long'], errors='coerce')
    df_all['year'] = df_all['year'].astype(int)
    df_all['month'] = df_all['month'].astype(int)

    columns_to_convert = ['stationID', 'nameTH', 'nameEN', 'areaTH', 'areaEN', 'stationType']
    for col in columns_to_convert:
        df_all[col] = df_all[col].astype(pd.StringDtype())

    df_all.drop_duplicates(inplace=True)
    df_all['PM25.aqi'] = df_all['PM25.aqi'].mask(df_all['PM25.aqi'] < 0, pd.NA)
    df_all['PM25.aqi'] = df_all.groupby('stationID')['PM25.aqi'].transform(lambda x: x.fillna(method='ffill'))
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')

    return df_all

# --- Streamlit Dashboard ---

# --- Style Enhancement ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f5;
    }
    .stApp {
        background: linear-gradient(120deg, #e0f7fa, #ffffff);
    }
    .block-container {
        padding-top: 2rem;
    }
    .stTitle > h1 {
        font-size: 2.2rem;
        color: #006064;
        border-bottom: 3px solid #00bcd4;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #ccc;
    }
    .stMetric label {
        font-size: 1rem;
        color: #333;
    }
    .stMetric div {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00796b;
    }
    .stButton button {
        background-color: #00bcd4;
        color: white;
        font-size: 1rem;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0097a7;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Real-Time Air Quality Dashboard', page_icon='🦄', layout='wide')
st.title("Air Quality Dashboard from LakeFS 🌎")

df = load_data()
if df.empty:
    st.stop()

thai_time = datetime.now(ZoneInfo("Asia/Bangkok"))
st.caption(f"อัปเดตล่าสุด: {thai_time.strftime('%Y-%m-%d %H:%M:%S')}")

with st.sidebar:
    st.title("Air4Thai Dashboard")
    st.header("⚙️ Settings")
    max_date = df['timestamp'].max().date()
    min_date = df['timestamp'].min().date()
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)
    station_name = df['nameTH'].dropna().unique().tolist()
    station_name.sort()
    station_name.insert(0, "ทั้งหมด")
    station = st.selectbox("Select Station", station_name)

df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
if station != "ทั้งหมด":
    df_filtered = df_filtered[df_filtered['nameTH'] == station]

if df_filtered.empty:
    st.warning("ไม่พบข้อมูลในช่วงเวลาหรือสถานีที่เลือก")
    st.stop()

# KPI Section
k1, k2, k3 = st.columns(3)
k1.metric("🌡️ ค่าเฉลี่ยคุณภาพ PM2.5 ในอากาศ", f"{df_filtered['PM25.aqi'].mean():.2f}")
k2.metric("🔥 ค่าเฉลี่ยระดับ PM2.5 ของประเทศไทย", f"{df_filtered['PM25.color_id'].mean():.2f}")
k3.metric("📍 พื้นที่ที่มีระดับ PM2.5 สูงสุด", df_filtered.groupby('areaTH')['PM25.aqi'].mean().idxmax())

# Visualization
fig_col1, fig_col2 = st.columns([1.2, 1.8], gap='medium')

if station == "ทั้งหมด":
    df_selected = df_filtered.copy()
    title = "PM2.5 AQI ของทุกสถานี"
else:
    df_selected = df_filtered[df_filtered['nameTH'] == station]
    title = f"PM2.5 AQI - สถานี {station}"

with fig_col1:
    df_map = df_selected.groupby(['stationID', 'nameTH', 'lat', 'long'], as_index=False)['PM25.aqi'].mean()
    fig_map = px.scatter_geo(
        df_map,
        lat='lat', lon='long', color='PM25.aqi',
        hover_name='nameTH',
        color_continuous_scale='Turbo',
        title='แผนที่สถานีตรวจวัด PM2.5 AQI ในประเทศไทย',
        projection='natural earth'
    )
    fig_map.update_geos(lataxis_range=[5, 21], lonaxis_range=[93, 110])
    fig_map.update_layout(template="plotly_dark", margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map)

with fig_col2:
    if station == "ทั้งหมด":
        top_5 = df_selected.groupby('nameTH')['PM25.aqi'].mean().nlargest(5).index
        df_top5 = df_selected[df_selected['nameTH'].isin(top_5)]
        fig = px.line(df_top5.sort_values("timestamp"), x='timestamp', y='PM25.aqi', color='nameTH', title=f"Top 5 สถานี AQI สูงสุด")
    else:
        fig = px.line(df_selected.sort_values("timestamp"), x='timestamp', y='PM25.aqi', title=title)
    fig.update_layout(xaxis_title='Time', yaxis_title='PM2.5 AQI')
    st.plotly_chart(fig)

# Classification Section
st.divider()
st.subheader("🤖 PM2.5 Level Classification")

def classify_aqi(value):
    if value < 50:
        return 'Low'
    elif value < 100:
        return 'Moderate'
    else:
        return 'High'

df_filtered['AQI_Class'] = df_filtered['PM25.aqi'].apply(classify_aqi)
X = df_filtered[['lat', 'long']].fillna(0)
y = df_filtered['AQI_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.markdown("### 🔍 Classification Report")
st.text(classification_report(y_test, y_pred))

st.markdown("### 📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Moderate', 'High'])
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Low', 'Moderate', 'High'],
    y=['Low', 'Moderate', 'High'],
    hoverongaps=False,
    colorscale='Blues'
))
fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(fig_cm)

st.markdown("### 📎 Distribution of Predicted Classes")
class_counts = pd.Series(y_pred).value_counts()
fig_pie = px.pie(values=class_counts.values, names=class_counts.index, title="Predicted AQI Class Distribution")
st.plotly_chart(fig_pie, use_container_width=True)