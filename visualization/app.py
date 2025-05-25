import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pyarrow.parquet as pq
import s3fs
import time
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



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
    lakefs_path = "s3://dsi321-record-air-quality/main/airquality.parquet/year=2025"
    data_list = fs.glob(f"{lakefs_path}/*/*/*/*")
    df_all = pd.concat(
        [pd.read_parquet(f"s3://{path}", engine="pyarrow", filesystem=fs) for path in data_list],
        ignore_index=True
    )

    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])  # ✅ convert timestamp to datetime
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

    return df_all

def filter_data(df, start_date, end_date, station):
    df_filtered = df.copy()

    # Filter by date
def filter_data(df, start_date, end_date, station):
    df_filtered = df.copy()

    df_filtered["date_only"] = df_filtered["timestamp"].dt.date  # ✅ convert to date for comparison

    df_filtered = df_filtered[
        (df_filtered['date_only'] >= start_date) &
        (df_filtered['date_only'] <= end_date)
    ]

    if station != "ทั้งหมด":
        df_filtered = df_filtered[df_filtered['nameTH'] == station]

    df_filtered = df_filtered[df_filtered['PM25.aqi'] >= 0]

    return df_filtered


def train_rf_model(df):
    features = ["lat", "long", "hour", "month"]  # ✅ ใช้ features ที่มีใน schema
    X = df[features]
    y = df["AQ_Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score, X_test, y_test


# --- Streamlit UI ---
# --- Page config ---
st.set_page_config(
    page_title="🌿 Air Quality Dashboard Thailand",
    layout="wide"
)
# --- Title section ---
st.markdown("""
<h1 style='font-size: 48px; color: #1a3c40;'>🌿 Air Quality Dashboard (Thailand 🇹🇭)</h1>
<p style='font-size: 16px; color: #5c5c5c; margin-top: -10px;'>
This dashboard displays near real-time PM2.5 air quality levels across Thailand using data from LakeFS and applies a machine learning model to classify air status.
</p>
""", unsafe_allow_html=True)

# 🔹 Layout: Sidebar Filter (ซ้าย) + Content (ขวา)
#st.header("🌬️ ML Air Quality Dashboard (Thailand 🇹🇭)")

left, right = st.columns([1, 4])

with left:
    st.markdown("## 🔍 Filter")
    df = load_data()
    stations = ["ทั้งหมด"] + sorted(df["nameTH"].unique())

    start_date = st.date_input("📅 Start Date", df['timestamp'].min().date())
    end_date = st.date_input("📅 End Date", df['timestamp'].max().date())
    station = st.selectbox("📍 Select Station", stations)

    # Filtered data
    df_filtered = filter_data(df, start_date, end_date, station)

with right:
    st.markdown("### 🤖 Air Quality Classification Model")
    st.markdown(
        "<div style='background-color:#f2f9ff; padding:20px; border-radius:10px;'>",
        unsafe_allow_html=True
    )

    if len(df_filtered) > 50:
        # สร้างคอลัมน์ AQ_Status ถ้ายังไม่มี
        if "AQ_Status" not in df_filtered.columns:
            df_filtered["AQ_Status"] = (df_filtered["PM25.aqi"] > 100).astype(int)

        # ลบแถวที่มีค่าว่างใน feature สำคัญ
        df_filtered = df_filtered.dropna(subset=["AQ_Status", "lat", "long", "hour", "month"])

        # เช็กว่ามี 2 class ให้เทรนไหม
        if df_filtered["AQ_Status"].nunique() < 2:
            st.warning("⚠️ ยังมีข้อมูลเพียงคลาสเดียว (Good หรือ Bad Air) ไม่สามารถเทรนโมเดลได้ค่ะ")
        else:
            try:
                model, acc, X_test, y_test = train_rf_model(df_filtered)
                pred = model.predict(X_test)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Model Accuracy", f"{acc:.2%}")
                with col2:
                    st.metric("📦 Records Used", len(df_filtered))

                st.markdown("#### 📋 Classification Report")

                report_dict = classification_report(y_test, pred, target_names=["Good Air", "Bad Air"], output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()

                st.dataframe(report_df.style.format({
                    "precision": "{:.2f}",
                    "recall": "{:.2f}",
                    "f1-score": "{:.2f}",
                    "support": "{:.0f}"
                }))


            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
    else:
        st.warning("⚠️ ข้อมูลมีน้อยเกินไป ลองเลือกวันที่ให้กว้างขึ้นเพื่อเทรนโมเดลนะคะ")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Section: Graphs with blue backgrounds -----
    st.header("📊 Air Quality Visualizations")

    # Graph row 1
    col1, col2 = st.columns([2, 2])
    with col1:
        st.markdown("""
        <div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px;">
        <h5 style='color:#004466'>📊 Daily Average PM2.5 (Bar Chart)</h5>
        """, unsafe_allow_html=True)

        # ➤ Group by date and average
        df_filtered["date_only"] = df_filtered["timestamp"].dt.date
        df_daily = df_filtered.groupby("date_only")["PM25.aqi"].mean().reset_index()

        # ➤ Plot bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(df_daily["date_only"].astype(str), df_daily["PM25.aqi"], color="#66b3ff")
        ax.set_ylabel("PM2.5 AQI")
        ax.set_xlabel("Date")
        ax.set_title("Average Daily PM2.5 AQI", fontsize=10)
        ax.axhline(100, color='red', linestyle='--', label='Danger Threshold (100)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)




    with col2:
        st.markdown("""<div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px;"><h5 style='color:#004466'>🧯 PM2.5 Heatmap (Hour × Day)</h5>""", unsafe_allow_html=True)
        if not df_filtered.empty:
            heatmap_data = df_filtered.copy()
            heatmap_data["dayname"] = heatmap_data["timestamp"].dt.day_name()
            heatmap_group = heatmap_data.groupby(["hour", "dayname"])["PM25.aqi"].mean().reset_index()
            heatmap_fig = px.density_heatmap(
                heatmap_group,
                x="dayname",
                y="hour",
                z="PM25.aqi",
                color_continuous_scale="OrRd",
                height=250
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Graph row 2
    col3, col4 = st.columns([2, 2])
    with col3:
        st.markdown("""<div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px;"><h5 style='color:#004466'>🗺️ PM2.5 Map View</h5>""", unsafe_allow_html=True)
        map_data = df_filtered[["lat", "long", "PM25.aqi", "nameTH"]].dropna()
        map_fig = px.scatter_mapbox(
            map_data,
            lat="lat",
            lon="long",
            size="PM25.aqi",
            color="PM25.aqi",
            color_continuous_scale="RdYlGn_r",
            hover_name="nameTH",
            zoom=5,
            height=400
        )
        map_fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(map_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("""<div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px;"><h5 style='color:#004466'>🎯 Proportion of Good vs Bad Air</h5>""", unsafe_allow_html=True)
        aq_count = df_filtered["AQ_Status"].value_counts().rename({0: "Good Air", 1: "Bad Air"})
        pie_fig = px.pie(
            names=aq_count.index,
            values=aq_count.values,
            title="Air Quality Status"
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
