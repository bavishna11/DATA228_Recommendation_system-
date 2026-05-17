"""Feature Engineering Module.
Replaces Notebook 3.
"""
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline
from src.utils.spark_write_helper import spark_write_parquet

def init_spark():
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"]
        ).decode().strip()
    except Exception:
        pass
    spark = SparkSession.builder \
        .appName("Amazon_FeatureEngineering") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    return spark

from src.utils.config_loader import load_config

def run_feature_engineering():
    print("Running Feature Engineering...")
    config, root_dir = load_config()
    spark = init_spark()
    clean_path = str(root_dir / config["data"]["clean_path"])
    abs_path = "file:///" + os.path.abspath(clean_path).replace("\\", "/")
    
    try:
        df = spark.read.parquet(abs_path)
    except Exception as e:
        print(f"Could not load clean data: {e}")
        return

    # 1. ID Encoding
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index", handleInvalid="skip")
    product_indexer = StringIndexer(inputCol="parent_asin", outputCol="product_id_index", handleInvalid="skip")

    # 2. NLP Pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features", minDocFreq=5)

    # 3. Assembler
    assembler = VectorAssembler(
        inputCols=["tfidf_features", "user_id_index", "product_id_index"],
        outputCol="features",
        handleInvalid="skip"
    )

    pipeline = Pipeline(stages=[user_indexer, product_indexer, tokenizer, remover, hashing_tf, idf, assembler])
    
    print("Fitting feature pipeline...")
    feature_model = pipeline.fit(df)
    engineered_df = feature_model.transform(df)

    # 4. Save
    out_dir = str(root_dir / config["data"]["features_path"])
    cols_to_save = ["user_id", "parent_asin", "rating", "user_id_index", "product_id_index", "tfidf_features", "features"]
    
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    spark_write_parquet(engineered_df, out_dir, cols=cols_to_save)
    print(f"Engineered features saved to {out_dir}")

    pipe_path = str(root_dir / config["model"]["feature_pipeline_path"])
    os.makedirs(os.path.dirname(pipe_path), exist_ok=True)
    feature_model.write().overwrite().save(pipe_path)
    print(f"Pipeline model saved to {pipe_path}")
    
    spark.stop()

if __name__ == "__main__":
    run_feature_engineering()
