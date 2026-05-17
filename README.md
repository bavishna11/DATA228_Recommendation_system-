# Amazon Product Recommendation & Review Intelligence Platform

![Project Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5%2B-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)

An end-to-end Big Data machine learning pipeline and interactive dashboard for the [McAuley-Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset. This project processes gigabytes of raw consumer data to generate actionable business intelligence, customer personalization profiles, and intelligent product recommendations.

---

## 🎯 Executive Summary

This platform bridges the gap between heavy, offline Big Data processing and low-latency, online serving. It ingests massive raw JSONL datasets, cleans and engineers features using distributed PySpark, trains multiple machine learning algorithms tracked by MLflow, and ultimately serves the finalized models and analytics via a robust FastAPI backend and an interactive Streamlit frontend.

### ✨ Key Features
*   **Customer Personalization (ALS):** Uses Collaborative Filtering (Alternating Least Squares) to predict how a specific user will rate products they haven't seen yet.
*   **Review Intelligence (NLP):** Natural Language Processing pipeline that tokenizes reviews, applies TF-IDF, and utilizes classification algorithms to assess customer sentiment.
*   **Business Dashboard:** Interactive metrics and visualizations (powered by Altair) exposing raw dataset statistics, rating distributions, and product popularity.
*   **Microservice Architecture:** Fully decoupled Data Pipeline and Serving Layers, containerized using Docker for instant portability.

---

## 🛠 Technology Stack

*   **Data Processing:** PySpark, PyArrow, Pandas
*   **Machine Learning:** Spark MLlib (ALS, Logistic Regression, Random Forest, Naive Bayes, K-Means)
*   **MLOps:** MLflow (Experiment tracking, Metric logging, Model registry), SQLite
*   **Backend API:** FastAPI, Uvicorn
*   **Frontend UI:** Streamlit, Altair
*   **DevOps & Infrastructure:** Docker, Docker Compose, Pytest

---

## 🏗 System Architecture

The project architecture strictly adheres to modern ML engineering best practices by separating the heavy "Offline" computation from the real-time "Online" serving.

### 1. Offline Data & Machine Learning Pipeline
*Executed on-demand when new data is available.*
1.  **Ingestion (`src/ingestion/data_loader.py`):** Converts massive raw `.jsonl` files into highly-optimized, compressed Parquet files.
2.  **EDA (`src/eda/eda.py`):** Calculates global rating distributions and identifies viral products.
3.  **Preprocessing (`src/preprocessing/cleaner.py`):** Handles missing values, enforces schema casting, and applies "Cold-Start Filtering" (removing users/items with < 5 interactions).
4.  **Feature Engineering (`src/features/feature_engineering.py`):** NLP processing (`StringIndexer`, `Tokenizer`, `StopWordsRemover`, `HashingTF`, `IDF`) to vectorize review text.
5.  **Model Training (`src/models/train.py`):** Conducts a model tournament. Evaluates ALS for recommendations and K-Means/Classifiers for review analysis. Logs all artifacts natively to MLflow.

### 2. Online Serving Layer
*Containerized web services that read pre-computed artifacts.*
1.  **FastAPI Backend:** Loads the Spark Pipeline models via `mlflow.pyfunc` and serves high-speed recommendation logic and dataset queries.
2.  **Streamlit Dashboard:** The user-facing application providing interactive filters, data tables, and dynamic charting.

---

## 🚀 Step-by-Step Setup & Execution Guide

### Step 1: Prerequisites
*   [Python 3.10+](https://www.python.org/downloads/)
*   [Java 17](https://adoptium.net/) (Required locally for PySpark execution)
*   [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Step 2: Download the Dataset
Download raw `.jsonl` category files from the [McAuley-Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset and place them in the root `data/` directory.

### Step 3: Run the Offline Pipeline
Trigger the automated, end-to-end PySpark pipeline to process your data and train the models.

```bash
# Make the script executable
chmod +x start_pipeline.sh

# Run the pipeline
./start_pipeline.sh
```
*Note: This process may take several minutes depending on the size of your `.jsonl` files. Upon completion, the `data/features/`, `models/`, and `mlruns/` directories will be populated.*

### Step 4: Boot the Serving Layer (Docker)
Once the pipeline has finished creating the artifacts, spin up the web services.

```bash
# Build and start the microservices in detached mode
docker-compose up --build -d

# Verify containers are running
docker ps
```

### Step 5: Access the Application
*   **Interactive Dashboard:** [http://localhost:8501](http://localhost:8501)
*   **FastAPI Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚙️ Configuration (Single Source of Truth)

All project paths, tracking URIs, and network ports are centrally managed. To modify where data is saved or what port the API runs on, edit the **`configs/config.yaml`** file. 

The `src/utils/config_loader.py` utility dynamically propagates these settings to every script in the repository, ensuring zero path-related bugs.

---

## 🧪 Testing

The repository includes an automated test suite to validate environment configurations and ensure all modules are syntactically sound before execution.

```bash
# Run the test suite
PYTHONPATH=. pytest tests/
```

---

## 📂 Project Directory Map

```text
.
├── PROJECT_REPORT.md         # Final documentation and results
├── README.md                 # Primary project documentation
├── Dockerfile                # Multi-service container definition (with Java JRE)
├── docker-compose.yml        # Orchestration for API & Dashboard
├── requirements.txt          # Python dependencies
├── start_pipeline.sh         # One-click script to run the offline data & ML pipeline
│
├── api/                      # Backend Serving Layer
│   └── main.py               # FastAPI application
│
├── configs/                  # Centralized Project Configuration (Single Source of Truth)
│   └── config.yaml           # Ports, tracking URIs, file paths
│
├── dashboard/                # Frontend Serving Layer
│   └── app.py                # Streamlit dashboard application
│
├── data/                     # Data Lake (Mounted to Docker)
│   ├── features/             # ML-ready engineered features
│   ├── processed/            # Cleaned & filtered data
│   └── raw/                  # Ingested parquets & jsonl
│
├── mlruns/                   # MLflow Tracking Database & Experiment Artifacts
│
├── models/                   # Serialized ML Artifacts
│   ├── als_recommendation_model/ 
│   └── feature_pipeline/     
│
├── src/                      # Core Machine Learning & Data Logic
│   ├── eda/eda.py            # Generates statistics and rating distributions
│   ├── features/feature_engineering.py # Runs NLP and vectorizes data
│   ├── ingestion/data_loader.py # Converts JSONL to Parquet
│   ├── models/train.py       # Trains ALS, Classification, and Clustering algorithms
│   ├── preprocessing/cleaner.py # Fills nulls and filters cold-starts
│   └── utils/config_loader.py # Configuration loader utility
│
└── tests/                    # Validation & Unit Tests
    └── test_pipeline.py      # Pytest suite for checking modules and configs
```
