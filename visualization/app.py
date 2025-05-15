import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import s3fs
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI
from langchain.prompts import PromptTemplate

# --- LakeFS Settings ---
lakefs_endpoint = os.getenv("LAKEFS_ENDPOINT", "http://lakefs-dev:8000")
ACCESS_KEY = os.getenv("LAKEFS_ACCESS_KEY")
SECRET_KEY = os.getenv("LAKEFS_SECRET_KEY")

fs = s3fs.S3FileSystem(
    key=ACCESS_KEY,
    secret=SECRET_KEY,
    client_kwargs={'endpoint_url': lakefs_endpoint}
)

@st.cache_data()
def load_data():
    path = "s3://air-quality/main/airquality.parquet/year=2025"
    files = fs.glob(f"{path}/*/*/*/*")
    df = pd.concat([pd.read_parquet(f"s3://{f}", engine="pyarrow", filesystem=fs) for f in files], ignore_index=True)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['PM25.aqi'] = df['PM25.aqi'].mask(df['PM25.aqi'] < 0, pd.NA)
    df['PM25.aqi'] = df.groupby('stationID')['PM25.aqi'].transform(lambda x: x.fillna(method='ffill'))
    return df.drop_duplicates()

# Typhoon API Setup
typhoon_token = "sk-Rl48oPMyO4lVARDGidyzc8tZLBQQxzNdxtXFQWJrDxJOx1j8"
client = OpenAI(
    api_key=typhoon_token,
    base_url='https://api.opentyphoon.ai/v1'
)

typhoon_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    คุณคือนักวิเคราะห์ข้อมูลสิ่งแวดล้อม สรุปสถานการณ์คุณภาพอากาศโดยใช้โครงสร้าง:
    
    {context}

    📌 Executive Summary
    🔍 Key Insights
    ⚠️ Policy Recommendations
    📈 Trend Forecast
    """
)

def generate_insight(summary):
    prompt = typhoon_prompt.format(context=summary)
    response = client.chat.completions.create(
        model="typhoon-v2-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )
    return response.choices[0].message.content

# --- Streamlit UI Setup ---
st.set_page_config("Air Quality Insights", layout="wide")
st.title("☁️ Interactive Air Quality Dashboard")

st.caption(f"Updated: {datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y-%m-%d %H:%M:%S')}")
df = load_data()

stations = sorted(df['nameTH'].dropna().unique().tolist())
stations.insert(0, "ทั้งหมด")

with st.container():
    col1, col2, col3 = st.columns([1, 1, 2])
    start_date = col1.date_input("📅 Start Date", df['timestamp'].min().date())
    end_date = col2.date_input("📅 End Date", df['timestamp'].max().date())
    station = col3.selectbox("🏢 Select Station", stations)

filtered = df.copy()
filtered = filtered[(filtered['timestamp'].dt.date >= start_date) & (filtered['timestamp'].dt.date <= end_date)]
if station != "ทั้งหมด":
    filtered = filtered[filtered['nameTH'] == station]

if filtered.empty:
    st.warning("ไม่พบข้อมูลในช่วงเวลาที่เลือก")
    st.stop()

# --- KPI Cards ---
k1, k2, k3 = st.columns(3)
k1.metric("🌡️ PM2.5 AQI เฉลี่ย", f"{filtered['PM25.aqi'].mean():.2f}")
k2.metric("📍 สถานีที่มีค่า AQI สูงสุด", filtered.groupby('nameTH')['PM25.aqi'].mean().idxmax())
k3.metric("🗓️ จำนวนข้อมูล", f"{len(filtered)} records")

# --- Charts ---
st.subheader("🌏 PM2.5 Map")
map_df = filtered.groupby(['stationID', 'nameTH', 'lat', 'long'], as_index=False)['PM25.aqi'].mean()
fig_map = px.scatter_geo(
    map_df,
    lat='lat', lon='long', color='PM25.aqi',
    hover_name='nameTH',
    title='ค่าเฉลี่ย PM2.5 รายสถานี',
    color_continuous_scale='Plasma',
    projection='natural earth'
)
st.plotly_chart(fig_map, use_container_width=True)

st.subheader("📈 AQI Timeline")
if station == "ทั้งหมด":
    top_stations = filtered.groupby('nameTH')['PM25.aqi'].mean().nlargest(5).index
    df_top = filtered[filtered['nameTH'].isin(top_stations)]
    fig_line = px.line(df_top.sort_values("timestamp"), x='timestamp', y='PM25.aqi', color='nameTH')
else:
    fig_line = px.line(filtered.sort_values("timestamp"), x='timestamp', y='PM25.aqi', title=f"แนวโน้ม AQI - {station}")
st.plotly_chart(fig_line, use_container_width=True)

# --- LLM Insight ---
st.divider()
st.subheader("💡 AI Insight Generator")

if st.button("Generate Insight with Typhoon AI"):
    summary = filtered.describe(include='all').to_string()
    with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
        insight = generate_insight(summary)
        st.success("วิเคราะห์สำเร็จแล้ว!")
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #eef2f6; border-left: 5px solid #4f46e5; border-radius: 8px;'>
        <h4>📊 ผลการวิเคราะห์</h4>
        <div>{insight}</div>
        </div>
        """, unsafe_allow_html=True)
