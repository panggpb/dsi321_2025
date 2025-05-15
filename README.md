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


## 🚀 Getting Started
### Prerequisites
-Python 3.9 or higher  
-pip package manager
### Installation
### 1.Clone the repository:
```git clone https://github.com/yourusername/dsi321_2025.git
cd dsi321_2025```
### 2.Create a virtual environment:
```python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate```
### 3.Install dependencies:
`pip install -r requirements.txt`
#### 4.Run the Streamlit application:
`streamlit run app/dashboard.py`
