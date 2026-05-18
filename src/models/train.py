"""Model Training & Tracking Script.
Replaces Notebooks 4 and 5.
"""
import os
import subprocess
import yaml
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    RegressionEvaluator,
    MulticlassClassificationEvaluator,
    ClusteringEvaluator,
)

from src.utils.config_loader import load_config

def init_spark():
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"]
        ).decode().strip()
    except Exception:
        pass # fallback to default if not mac

    spark = SparkSession.builder \
        .appName("Amazon_Model_Training") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    return spark

def train_als(spark, features_path, model_out_dir):
    print("--- Training ALS Model ---")
    abs_path = "file:///" + os.path.abspath(features_path).replace("\\", "/")
    try:
        als_all = spark.read.parquet(abs_path)
    except Exception as e:
        print(f"Skipping ALS, could not load data: {e}")
        return

    als_all = als_all.select(
        col("user_id_index").cast("float"),
        col("product_id_index").cast("float"),
        col("rating").cast("float"),
    ).na.drop(subset=["user_id_index", "product_id_index", "rating"])

    (train_als, test_als) = als_all.randomSplit([0.8, 0.2], seed=42)
    train_als.cache()
    
    with mlflow.start_run(run_name="ALS_Collaborative_Filtering"):
        als = ALS(
            maxIter=10,
            regParam=0.1,
            rank=10,
            userCol="user_id_index",
            itemCol="product_id_index",
            ratingCol="rating",
            coldStartStrategy="drop",
        )
        model = als.fit(train_als)
        predictions = model.transform(test_als)
        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        
        mlflow.log_param("rank", als.getRank())
        mlflow.log_param("regParam", als.getRegParam())
        mlflow.log_param("maxIter", als.getMaxIter())
        mlflow.log_metric("rmse", float(rmse))
        mlflow.spark.log_model(model, "als_model")
        
        os.makedirs(model_out_dir, exist_ok=True)
        model_path = os.path.join(model_out_dir, "als_recommendation_model")
        model.write().overwrite().save(model_path)
        print(f"Saved ALS to {model_path}")
        print(f"ALS RMSE (test): {rmse:.4f}")
    
    train_als.unpersist()

def train_classification_and_clustering(spark, clean_data_path, pipeline_model_path):
    print("--- Training NLP & Clustering Models ---")
    ML_SAMPLE_FRACTION = 0.12
    abs_path = "file:///" + os.path.abspath(clean_data_path).replace("\\", "/")
    
    try:
        raw_df = spark.read.parquet(abs_path)
        if ML_SAMPLE_FRACTION < 1.0:
            raw_df = raw_df.sample(withReplacement=False, fraction=ML_SAMPLE_FRACTION, seed=42)
        
        pipeline_model = PipelineModel.load(pipeline_model_path)
    except Exception as e:
        print(f"Skipping NLP/Clustering, could not load data/pipeline: {e}")
        return

    feat_df = pipeline_model.transform(raw_df)
    feat_df = feat_df.withColumn(
        "label", when(col("rating") >= 4.0, 1.0).otherwise(0.0)
    )

    cls_all = feat_df.select("tfidf_features", "label").na.drop()
    (train_cls, test_cls) = cls_all.randomSplit([0.8, 0.2], seed=43)
    train_cls.cache()

    km_all = feat_df.select("features").na.drop()
    (train_km, test_km) = km_all.randomSplit([0.8, 0.2], seed=44)
    train_km.cache()

    acc_eval = MulticlassClassificationEvaluator(
        predictionCol="prediction", labelCol="label", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        predictionCol="prediction", labelCol="label", metricName="f1"
    )

    # 1. Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression_TFIDF"):
        lr = LogisticRegression(
            featuresCol="tfidf_features", labelCol="label", maxIter=40, family="multinomial"
        )
        lr_model = lr.fit(train_cls)
        lr_pred = lr_model.transform(test_cls)
        mlflow.log_metric("accuracy", float(acc_eval.evaluate(lr_pred)))
        mlflow.log_metric("f1", float(f1_eval.evaluate(lr_pred)))
        mlflow.log_param("maxIter", 40)
        mlflow.spark.log_model(lr_model, "lr_model")
        print("Trained LR.")

    # 2. Random Forest
    with mlflow.start_run(run_name="RandomForest_TFIDF"):
        rf = RandomForestClassifier(
            featuresCol="tfidf_features", labelCol="label", numTrees=40, maxDepth=12, seed=42
        )
        rf_model = rf.fit(train_cls)
        rf_pred = rf_model.transform(test_cls)
        mlflow.log_metric("accuracy", float(acc_eval.evaluate(rf_pred)))
        mlflow.log_metric("f1", float(f1_eval.evaluate(rf_pred)))
        mlflow.log_param("numTrees", 40)
        mlflow.log_param("maxDepth", 12)
        mlflow.spark.log_model(rf_model, "rf_model")
        print("Trained RF.")

    # 3. Naive Bayes
    with mlflow.start_run(run_name="NaiveBayes_TFIDF"):
        nb = NaiveBayes(
            featuresCol="tfidf_features", labelCol="label", smoothing=1.0, modelType="multinomial"
        )
        nb_model = nb.fit(train_cls)
        nb_pred = nb_model.transform(test_cls)
        mlflow.log_metric("accuracy", float(acc_eval.evaluate(nb_pred)))
        mlflow.log_metric("f1", float(f1_eval.evaluate(nb_pred)))
        mlflow.log_param("smoothing", 1.0)
        mlflow.spark.log_model(nb_model, "nb_model")
        print("Trained NB.")

    train_cls.unpersist()

    # 4. K-Means
    km_eval = ClusteringEvaluator(
        featuresCol="features", predictionCol="prediction", metricName="silhouette", distanceMeasure="squaredEuclidean"
    )
    with mlflow.start_run(run_name="KMeans_features"):
        kmeans = KMeans(featuresCol="features", k=5, seed=42, maxIter=30)
        km_model = kmeans.fit(train_km)
        km_pred = km_model.transform(test_km)
        silhouette = float(km_eval.evaluate(km_pred))
        mlflow.log_metric("silhouette", silhouette)
        mlflow.log_param("k", 5)
        mlflow.log_param("maxIter", 30)
        mlflow.spark.log_model(km_model, "kmeans_model")
        print(f"K-Means silhouette: {silhouette:.4f}")

    train_km.unpersist()

def main():
    config, root_dir = load_config()
    
    _mlruns = os.path.abspath(os.path.join(str(root_dir), "mlruns"))
    mlflow.set_tracking_uri("file:" + _mlruns)
    mlflow.set_experiment(config["model"]["experiment_name"])
    
    spark = init_spark()
    
    # 1. Train ALS
    features_path = str(root_dir / config["data"]["features_path"])
    model_out_dir = str(root_dir / "models")
    train_als(spark, features_path, model_out_dir)
    
    # 2. Train Classifiers
    clean_path = str(root_dir / config["data"]["clean_path"])
    pipe_path = str(root_dir / config["model"]["feature_pipeline_path"])
    train_classification_and_clustering(spark, clean_path, pipe_path)
    
    spark.stop()
    print("All models trained successfully!")

if __name__ == "__main__":
    main()
