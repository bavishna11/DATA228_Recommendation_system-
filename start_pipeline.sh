#!/usr/bin/env bash
set -e

echo "Running end-to-end pipeline..."

# 1. Data Ingestion & Profiling
python src/ingestion/data_loader.py
python src/eda/eda.py

# 2. Data Cleaning
python src/preprocessing/cleaner.py

# 3. Feature Engineering
python src/features/feature_engineering.py

# 4. Model Training & Tracking
python src/models/train.py

echo "Pipeline completed successfully!"
