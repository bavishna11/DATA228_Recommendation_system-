# Pipeline Run Evidence & MLflow Artifacts

## Run Summary
- **Dataset**: Amazon Reviews Parquet (~4.6M records)
- **Model Training**: ALS Collaborative Filtering, Logistic Regression, Random Forest, Naive Bayes, K-Means
- **Pipeline Script**: `start_pipeline.sh` (runs `src/` modules end-to-end)

## Model Metrics (Latest Run)
- **ALS RMSE (test)**: 1.4381
- **Classification & Clustering**: Logged to MLflow (LR, RF, NB accuracy + K-Means silhouette)

## MLflow Tracking
All runs have been successfully tracked in `mlruns/`.
You can view the MLflow UI by running:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Test Results
```
tests/test_pipeline.py::test_config_loads PASSED
tests/test_pipeline.py::test_pipeline_modules_importable PASSED
tests/test_pipeline.py::test_directory_structure PASSED
tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_predict_sentiment_positive PASSED
tests/test_api.py::test_predict_sentiment_negative PASSED
tests/test_api.py::test_predict_sentiment_neutral PASSED
============================== 7 passed ===============================
```

## Evidence
- `models/als_recommendation_model/` contains the exported ALS Spark model.
- `models/feature_pipeline/` contains the trained NLP feature pipeline.
- `mlruns/` contains parameter, metric, and artifact metadata.
- `test_report.xml` contains JUnit XML test results.
- Docker containers `recommendation_api` and `recommendation_dashboard` serve the online layer.
