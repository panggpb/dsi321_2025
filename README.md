# Automated ML Pipelines for PM2.5 Forecasting with LakeFS + Prefect

## Project Overview
This project is part of the DSI321: BIG DATA INFRASTRUCTURE course. It presents a real-time air quality analytics system focused on PM2.5 levels in Thailand. The data pipeline is orchestrated using Prefect.io, with storage in LakeFS and Parquet format.

A classification model using Random Forest is applied to predict whether air quality is "Good" or "Bad". The results are visualized through an interactive Streamlit dashboard, which includes time-series plots, maps, and classification outputs.

## Introduction
This project is all about building a system that can automatically collect, process, and analyze real-time air quality data — especially PM2.5 — from monitoring stations across Thailand. Our goal is to make this information easy to access, understand, and use through a clean, interactive dashboard.

By using real data from the Air4Thai API (provided by Thailand’s Pollution Control Department), we can monitor the situation hour by hour with reliable and up-to-date information.

To make the pipeline work smoothly, we used Prefect 3 to automate every step — from fetching the data to cleaning, storing, and analyzing it. The data is saved in Parquet format and versioned through LakeFS, so we can track changes and keep everything organized.

For the analysis part, we trained a Random Forest Classifier to help classify whether the air quality is “Good” or “Bad,” based on time and location. This makes it easier for people to see when and where the air might be risky.

Finally, everything comes together in a Streamlit dashboard that we built to show trends, predictions, and maps in a way that’s interactive and user-friendly. Whether you’re a policymaker, a local resident, or just someone concerned about the air you breathe, this tool is designed to help you stay informed.  

## Objective  
- Data Collection: Gather real-time PM2.5 data from multiple stations across Thailand.
- Data Processing: Clean and preprocess the data to ensure accuracy and consistency.
- Visualization: Create interactive charts and maps to display air quality trends.
- Machine Learning: Implement a classification model to predict air quality status.

## Visualizations

The dashboard includes the following visual components:

- Line Chart: Displays PM2.5 AQI trends over time.
- Map View: Shows station locations with color-coded AQI levels.
- Hourly Heatmap: Illustrates average AQI values across different hours.
- Pie Chart: Represents the proportion of 'Good' vs. 'Bad' air quality statuses.

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



