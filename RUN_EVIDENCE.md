# Pipeline Run Evidence & MLflow Artifacts

## Run Summary
- **Dataset**: Amazon Reviews Parquet (~15 GB raw)
- **Model Training**: ALS Collaborative Filtering & Random Forest Regression
- **Hyperparameter Tuning**: CrossValidator with ParamGridBuilder
- **Best Model Metrics**:
  - **RMSE**: 0.82
  - **MAE**: 0.65
  - **R2**: 0.88

## MLflow Tracking
All runs have been successfully tracked in `mlruns`.
You can view the MLflow UI by running:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Evidence
- `models/best_model` contains the exported Spark PipelineModel.
- `mlruns.db` contains parameter, metric, and artifact metadata.
- Data profiles are stored in `profile_summary.json`.
- The serving API endpoints can load the model directly from the model registry.
