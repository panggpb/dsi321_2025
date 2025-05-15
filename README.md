# Automated ML Pipelines for PM2.5 Forecasting with LakeFS + Prefect

# Project Overview
This project is part of the DSI321: BIG DATA INFRASTRUCTURE course. It presents a real-time air quality analytics system focused on PM2.5 levels in Thailand. The data pipeline is orchestrated using Prefect.io, with storage in LakeFS and Parquet format.

A classification model using Random Forest is applied to predict whether air quality is "Good" or "Bad". The results are visualized through an interactive Streamlit dashboard, which includes time-series plots, maps, and classification outputs.

# Introduction
This project focuses on building a robust end-to-end system for real-time air quality data orchestration, forecasting, and visualization using technologies such as Prefect 3, Docker, and GitHub. It plays a critical role in supporting environmental surveillance, public health initiatives, and urban planning by delivering timely and actionable insights into air quality trends and potential risks, particularly in the Rangsit area of Pathum Thani, Thailand.

The system leverages the Air4Thai API, which provides hourly PM2.5 concentration data collected from automated air quality monitoring stations across Thailand. The API is maintained by Thailand’s Pollution Control Department, ensuring authoritative and up-to-date environmental information.
