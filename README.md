# Real-Time PM 2.5 Air Quality Data Pipeline with LakeFS + Prefect
**Author**: Intita Pitpongkul  
**Course**: DSI321 – Big Data Infrastructure  
**Institution**: Data Science and Innovation, Thammasat University

## Project Overview
This project is part of the DSI321: BIG DATA INFRASTRUCTURE course. It presents a real-time air quality analytics system focused on PM2.5 levels in Thailand. The data pipeline is orchestrated using Prefect.io, with storage in LakeFS and Parquet format.

A classification model using Random Forest is applied to predict whether air quality is "Good" or "Bad". The results are visualized through an interactive Streamlit dashboard, which includes time-series plots, maps, and classification outputs.

## Introduction
This project focuses on building a system that automatically collects, processes, and analyzes real-time air quality data—especially PM2.5—from monitoring stations across Thailand. The goal is to make this information accessible, understandable, and actionable through a clean, interactive dashboard.

Data is retrieved from the Air4Thai API, provided by Thailand’s Pollution Control Department, allowing for hour-by-hour monitoring using reliable and up-to-date sources.

The pipeline is automated using Prefect 3, covering every step from data ingestion to cleaning, transformation, and storage. All data is saved in Parquet format and version-controlled using LakeFS, enabling reproducibility, traceability, and organized data management.

For analysis, a Random Forest Classifier is trained to classify air quality as either “Good” or “Bad,” based on temporal and geographic features. This classification helps identify areas and times with potentially hazardous air quality conditions.

Insights are delivered through an interactive Streamlit dashboard, which displays trends, predictions, maps, and other visualizations. The dashboard is designed to support decision-makers, local communities, and the general public in staying informed about current air quality conditions. 

## Objective  
- Data Collection: Gather real-time PM2.5 data from multiple stations across Thailand.
- Data Processing: Clean and preprocess the data to ensure accuracy and consistency.
- Visualization: Create interactive charts and maps to display air quality trends.
- Machine Learning: Implement a classification model to predict air quality status.

# Dataset Preparation and Integration  
The system automatically retrieves near real-time PM2.5 air quality data from the Air4Thai API, maintained by Thailand’s Pollution Control Department. Once ingested, the dataset undergoes several processing steps to ensure usability and structure:  

<img width="1202" alt="Screenshot 2568-05-25 at 4 51 15 PM" src="https://github.com/user-attachments/assets/85b0fcbf-6087-4a9b-88f3-9c3ee69a2185" />  
- Data Cleaning: Invalid readings (e.g., negative PM2.5 values) are removed or forward-filled based on station history.
- Time-based Partitioning: Data is organized into a hierarchical folder structure by year/month/day/hour for efficient querying.
- Parquet Storage on lakeFS: Cleaned records are saved in partitioned Parquet format within lakeFS, enabling version control and reproducibility.
- Schema Validation: A defined schema (schema.md) is used to validate data types and column names, minimizing inconsistencies and preserving pipeline stability over time.

This pipeline ensures that the data is always well-structured, traceable, and ready for downstream use in dashboards and modeling.

# Data Schema
```json
{
  "columns": [
    "timestamp", "stationID", "nameTH", "nameEN", "areaTH",
    "areaEN", "stationType", "lat", "long", "PM25.color_id",
    "PM25.aqi", "year", "month", "day", "hour"
  ],
  "types": [
    "datetime64[ns]", "string", "string", "string", "string", 
    "string", "string", "float64", "float64", "int64",  
    "float64", "int64", "int64", "int32", "int32"
  ],
  "key_columns": [
    "timestamp", "stationID", "lat", "long", "PM25.aqi"
  ]
}
```

## Field Descriptions
|Column Name|	Description|
|-----------|---------------------------------------------------|
|timestamp  |	ISO format timestamp of data collection|
|stationID  |	Unique code identifying each monitoring station|
|nameTH     |	Station name in Thai|
|nameEN     |	Station name in English|
|areaTH     |	Area name in Thai|
|areaEN     |	Area name in English|
|stationType|	Type of station (e.g., general, roadside)|
|lat, long  |	Geographic coordinates of the station|
|PM25.color_id|	Visual indicator for AQI level coloring|
|PM25.aqi|	PM2.5 Air Quality Index (numeric value)|
|year, month, day, hour|	Timestamp components extracted for temporal analysis|  

**Key columns** (timestamp, stationID, lat, long, PM25.aqi) are mandatory for quality assurance and must be non-null. Schema validation ensures data completeness and correct typing for reliable modeling and visualization.

## Getting Started
### Prerequisites
-Python 3.9 or higher  
-pip package manager
### Installation
### 1.Clone the repository:
```
git clone https://github.com/yourusername.git
cd this-repo-folder
```
### 2.Start Docker Services:
```
docker-compose up -d --build
```
After this, you can access:  
Prefect Dashboard : http://localhost:4200  
JupyterLab : http://localhost:8888  
LakeFS : http://localhost:8001 (changed from default 8000)  
Stramlit : http://localhost:8501
### 3.Deploy Prefect Flow:
```
python src/pipeline.py deploy
python deploy.py
```

## Visualizations

This dashboard presents near real-time PM2.5 air quality data across Thailand through interactive and interpretable visualizations built using Matplotlib and Plotly via Streamlit. These visual tools help translate complex environmental data into actionable insights for citizens and policymakers.:

- Line Chart: Displays PM2.5 AQI trends over time.
- Map View: Shows station locations with color-coded AQI levels.
- Hourly Heatmap: Illustrates average AQI values across different hours.
- Pie Chart: Represents the proportion of 'Good' vs. 'Bad' air quality statuses.
- Dropdown filters (Station / Date-Time)

# 🔍 Filtering Options: Province & Station  
Users can select a Province and specific Monitoring Station via dropdown filters in the sidebar. All charts update dynamically based on these selections, allowing for customized exploration.  
Insight: For example, selecting “Bangkok” immediately filters the dashboard to show data trends from inner-city stations, helping users monitor local air quality.  
<img width="275" alt="Screenshot 2568-05-25 at 4 37 05 PM" src="https://github.com/user-attachments/assets/637e8a8b-5a76-489f-b9fc-5591d45cd32b" />  

# ✅ KPI Scorecards (Top Summary)  

At the top of the main dashboard, users see an overview of:
	•	Model Accuracy from the Random Forest classification
	•	Number of Records Used in training
These values automatically update with filtered data, reflecting the dashboard’s real-time adaptability.  

Insight: The model achieved ~95% accuracy on filtered data, distinguishing between “Good” and “Bad” air quality based on PM2.5 > 100 µg/m³.  
<img width="664" alt="Screenshot 2568-05-25 at 4 39 11 PM" src="https://github.com/user-attachments/assets/0b1ba769-2b34-41e2-a6df-31b4afdb8ad9" />  

# 📊 Daily Average PM2.5 (Bar Chart)  

Displays the average PM2.5 level per day, plotted as vertical bars. A red dashed line marks the hazardous threshold at 100 µg/m³.  

Insight: Users can detect daily pollution trends, and visually compare spikes across different days. For instance, certain weekends may show lower PM2.5 due to reduced traffic.  

<img width="542" alt="Screenshot 2568-05-25 at 4 40 12 PM" src="https://github.com/user-attachments/assets/4d01403a-eb06-436b-9af0-01d0aa00b945" />  

# 🧯 PM2.5 Heatmap (Hour × Day)  

A density heatmap shows PM2.5 variation across hours and weekdays.
	•	X-axis: Day of week
	•	Y-axis: Hour of day
	•	Color: Average PM2.5 level  
 
Insight: Morning and evening rush hours tend to have higher PM2.5, especially on weekdays like Monday and Friday.  
<img width="515" alt="Screenshot 2568-05-25 at 4 41 07 PM" src="https://github.com/user-attachments/assets/b51b12e2-f5c9-4771-a06f-0640052a343e" />  

# 🗺️ Station Map View (Mapbox)  

Each station is plotted on an interactive map, where:
	•	Marker size and color represent PM2.5 level
	•	Hover reveals station name and exact AQI reading  
 
Insight: High-PM2.5 stations (e.g., near industrial zones or highways) are easily spotted. This spatial insight supports targeted environmental intervention.  
<img width="524" alt="Screenshot 2568-05-25 at 4 41 53 PM" src="https://github.com/user-attachments/assets/32d56266-fc35-467e-b49c-35277e4f8961" />  

# 🎯 Pie Chart: Proportion of Air Quality  

This chart summarizes how many records fall under:
	•	Good Air (PM2.5 ≤ 100)
	•	Bad Air (PM2.5 > 100)  
 
Insight: Users get an immediate sense of the air quality distribution during their selected date range.  

# 🤖 Machine Learning: Air Quality Classification  

The project uses a Random Forest Classifier to classify whether a given record’s PM2.5 level is “Good” or “Bad” using features like:
	•	Latitude / Longitude
	•	Hour of day
	•	Month  
 
Insight: The model provides strong generalization with an accuracy typically above 90%, helping users anticipate and identify risk-prone hours and locations.  

<img width="1089" alt="Screenshot 2568-05-25 at 4 43 04 PM" src="https://github.com/user-attachments/assets/0ea6fcba-8260-4715-8805-7ce2d9bdbf0e" />  


## Technologies Used

This project integrates a range of modern technologies to achieve real-time data processing, visualization, and machine learning classification:

|Category|	Tools / Frameworks|
|--------|--------------------|
|Programming|	Python 3.9|
|Data Handling|	pandas, numpy|
|Data Storage|	Parquet format (via PyArrow), LakeFS (object versioning)|
|Visualization|	Streamlit, matplotlib, seaborn, plotly, pydeck|
|Machine Learning| scikit-learn (Random Forest Classifier)|
|Web Dashboard|	Streamlit|
|File| System / API	s3fs (S3 access for LakeFS integration)|
|Version Control|	Git, GitHub|


# Conclusion  

This project demonstrates the successful development of a real-time air quality monitoring pipeline using open-source technologies including Prefect, lakeFS, Docker, and Streamlit. By integrating automated data ingestion with machine learning classification and interactive dashboards, it enables:
	•	Daily monitoring of PM2.5 levels across Thailand
	•	Location-based insights through geographic and temporal visualizations
	•	Air quality classification to distinguish between “Good” and “Bad” conditions using Random Forest
	•	Scalable data management via partitioned Parquet format and versioned storage with lakeFS  
 
The system offers not only technical robustness, but also real-world value — helping researchers, citizens, and decision-makers better understand air pollution patterns and take timely action.  

As air quality continues to be a pressing issue, especially in urban and industrial regions, this solution provides a foundation for sustainable environmental data platforms.  

