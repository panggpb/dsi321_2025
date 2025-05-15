import os
import pandas as pd
import streamlit as st
import plotly.express as px
import s3fs

# --- Page Config ---
st.set_page_config(page_title="PM2.5 Dashboard", layout="wide")

# --- Load Custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
menu = st.sidebar.radio("Navigation", ["Dashboard", "ML Clustering Map View","Raw Data"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**User:** You\nVersion: 1.0")
st.sidebar.button("Logout")

# --- S3 Access Setup ---
fs = s3fs.S3FileSystem(
    key=os.getenv("LAKEFS_ACCESS_KEY"),
    secret=os.getenv("LAKEFS_SECRET_KEY"),
    client_kwargs={"endpoint_url": os.getenv("LAKEFS_ENDPOINT").replace("http", "https")},
    use_ssl=False
)

@st.cache_data(ttl=2400)
def load_data():
    path = "s3://dsi321-air-quality/main/airquality.parquet/year=2025"
    files = fs.glob(f"{path}/*/*/*/*")
    df = pd.concat([pd.read_parquet(f"s3://{f}", filesystem=fs) for f in files], ignore_index=True)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    df['PM25.value'] = pd.to_numeric(df['PM25.value'], errors='coerce')
    df['province'] = df['areaEN'].str.extract(r",\s*([^,]+)$")[0].str.strip()
    return df.dropna(subset=['lat', 'long', 'PM25.value'])

df = load_data()

# --- Title Header ---
st.title("🌫️ PM2.5 Air Quality Dashboard")

# --- Filters (used globally) ---
col1, col2 = st.columns(2)

with col1:
    selected_province = st.selectbox("📍 Province", ["All"] + sorted(df['province'].dropna().unique().tolist()), key="province")

with col2:
    if selected_province == "All":
        station_options = sorted(df["nameEN"].dropna().unique().tolist())
    else:
        station_options = sorted(df[df["province"] == selected_province]["nameEN"].dropna().unique().tolist())

    selected_station = st.selectbox("🏛️ Station", ["All"] + station_options, key="station")

# --- Apply Filter ---
df_filtered = df.copy()
if selected_province != "All":
    df_filtered = df_filtered[df_filtered["province"] == selected_province]
if selected_station != "All":
    df_filtered = df_filtered[df_filtered["nameEN"] == selected_station]

# --- Dashboard ---
if menu == "Dashboard":
    k1, k3, k4 = st.columns(3)

    with k1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{len(df_filtered)}</div>
            </div>
        """, unsafe_allow_html=True)

    df_valid = df_filtered.dropna(subset=["PM25.value", "province", "nameEN"])

    if not df_valid.empty:
        max_row = df_valid.loc[df_valid["PM25.value"].idxmax()]
        min_row = df_valid.loc[df_valid["PM25.value"].idxmin()]

        with k3:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Max PM2.5</div>
                    <div class="metric-value">{max_row['PM25.value']:.1f}</div>
                    <div class="metric-delta">📍 {max_row['nameEN']}, {max_row['province']}</div>
                </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Min PM2.5</div>
                    <div class="metric-value">{min_row['PM25.value']:.1f}</div>
                    <div class="metric-delta">📍 {min_row['nameEN']}, {min_row['province']}</div>
                </div>
            """, unsafe_allow_html=True)

    # --- Charts ---
    st.markdown("---")
    st.subheader("📊 PM2.5 Insights")

    c1, c2 = st.columns(2)

    with c1:
        if selected_province == "All":
            top10 = df_filtered.groupby("province", as_index=False)["PM25.value"].max()
            y_col = "province"
            chart_title = "Top 10 Provinces by Max PM2.5"
        else:
            top10 = df_filtered.groupby("nameEN", as_index=False)["PM25.value"].max()
            y_col = "nameEN"
            chart_title = f"Top 10 Stations in {selected_province} by Max PM2.5"

        top10 = top10.sort_values("PM25.value", ascending=False).head(10)
        fig1 = px.bar(
            top10,
            x="PM25.value",
            y=y_col,
            orientation="h",
            title=chart_title,
            labels={"PM25.value": "PM2.5", y_col: "Location"},
            color="PM25.value",
            color_continuous_scale="reds"
        )
        fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        df_box = df_filtered.dropna(subset=["PM25.value", "province"])
    
        if not df_box.empty:
            # Step 1: Find top 5 provinces by max PM2.5
            top5_provinces = (
                df_box.groupby("province", as_index=False)["PM25.value"]
                .max()
                .sort_values("PM25.value", ascending=False)
                .head(5)["province"]
            )

            # Step 2: Filter only top 5 provinces
            df_box_top5 = df_box[df_box["province"].isin(top5_provinces)]

            # Step 3: Plot
            fig2 = px.box(
                df_box_top5,
                x="province",
                y="PM25.value",
                points="outliers",
                title="Top 5 Provinces by PM2.5 Distribution",
                labels={"province": "Province", "PM25.value": "PM2.5"},
            )
            fig2.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No valid PM2.5-province data to plot.")

# --- ML View ---
elif menu == "ML Clustering Map View":
    st.title("ML View: K-Means Clustering on PM2.5")
    st.markdown("Each color represents a cluster of stations based on similar PM2.5 levels using K-Means (n=3)")

    df_ml = df_filtered.dropna(subset=["PM25.value", "lat", "long", "nameEN"]).copy()
    df_ml = df_ml[df_ml["PM25.value"] > 0]  # 🚨 Filter out invalid values

    if df_ml.empty:
        st.warning("No data available for clustering.")
    else:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        df_ml["cluster"] = kmeans.fit_predict(df_ml[["PM25.value"]])

        # Sort clusters by average PM2.5 and label them
        cluster_means = df_ml.groupby("cluster")["PM25.value"].mean().sort_values().reset_index()
        cluster_mapping = {old: new for new, old in enumerate(cluster_means["cluster"])}
        df_ml["cluster"] = df_ml["cluster"].map(cluster_mapping)
        cluster_names = {0: "Low PM2.5", 1: "Moderate PM2.5", 2: "High PM2.5"}
        df_ml["cluster_label"] = df_ml["cluster"].map(cluster_names)

        # 💡 Explanation section
        st.markdown("### 🧠 Clustering Interpretation")
        cluster_summary = (
            df_ml.groupby("cluster_label")["PM25.value"]
            .agg(["count", "min", "max", "mean"])
            .rename(columns={
                "count": "Stations",
                "min": "Min PM2.5",
                "max": "Max PM2.5",
                "mean": "Average PM2.5"
            })
            .reset_index()
        )
        st.dataframe(cluster_summary.style.format({
            "Min PM2.5": "{:.2f}",
            "Max PM2.5": "{:.2f}",
            "Average PM2.5": "{:.2f}"
        }))

        st.markdown("""
        **Interpretation of Clusters**  
        - 🟢 **Low PM2.5**: Stations with relatively clean air.
        - 🔵 **Moderate PM2.5**: Stations with noticeable but not critical pollution.
        - 🔴 **High PM2.5**: Stations with high pollution levels that may pose health risks.

        Clustering is performed using **K-Means** based on the `PM2.5` value across stations.  
        The algorithm automatically finds natural groupings of pollution levels into 3 categories.
        """)

        # 📍 Plot
        center_lat = df_ml["lat"].mean()
        center_lon = df_ml["long"].mean()

        fig = px.scatter_mapbox(
            df_ml,
            lat="lat",
            lon="long",
            color="cluster_label",
            size="PM25.value",
            hover_name="nameEN",
            mapbox_style="open-street-map",
            center={"lat": center_lat, "lon": center_lon},  # ✅ เพิ่ม center ตรงนี้
            zoom=6,  # ✅ ใส่แค่ครั้งเดียว
            title="📍 K-Means Clustering of PM2.5 Stations by Pollution Level",
            color_discrete_map={
                "Low PM2.5": "green",
                "Moderate PM2.5": "blue",
                "High PM2.5": "red"
            }
        )


        st.plotly_chart(fig, use_container_width=True)


# --- Raw Data ---
elif menu == "Raw Data":
    st.title("📄 Raw Data Table")
    st.dataframe(df_filtered)
    st.download_button("Download CSV", data=df_filtered.to_csv(index=False), file_name="pm25_filtered.csv")