"""Data Cleaning & Preprocessing Module.
Replaces Notebook 2.
"""
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def init_spark():
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"]
        ).decode().strip()
    except Exception:
        pass
    spark = SparkSession.builder \
        .appName("Amazon_Preprocessing") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()
    return spark

from src.utils.config_loader import load_config

def run_cleaning():
    print("Running Data Cleaning...")
    config, root_dir = load_config()
    spark = init_spark()
    parquet_path = str(root_dir / config["data"]["raw_path"])
    abs_path = "file:///" + os.path.abspath(parquet_path).replace("\\", "/")
    
    try:
        df = spark.read.parquet(abs_path)
    except Exception as e:
        print(f"Could not load data for cleaning: {e}")
        return

    # 1. Fill nulls
    df_clean = df.fillna({
        "rating": "3",
        "title": "Unknown",
        "text": "",
        "helpful_vote": 0,
        "verified_purchase": False
    })
    
    # 2. Drop rows with no user_id or parent_asin
    df_clean = df_clean.dropna(subset=["user_id", "parent_asin", "rating"])
    
    # 3. Cast rating to float
    df_clean = df_clean.withColumn("rating", col("rating").cast("float"))

    # 4. Cold-Start Filtering (min 5 reviews)
    user_counts = df_clean.groupBy("user_id").count().withColumnRenamed("count", "user_review_count")
    product_counts = df_clean.groupBy("parent_asin").count().withColumnRenamed("count", "product_review_count")

    df_filtered = df_clean \
        .join(user_counts, "user_id") \
        .join(product_counts, "parent_asin") \
        .filter((col("user_review_count") >= 5) & (col("product_review_count") >= 5))

    # 5. Save output
    out_dir = str(root_dir / config["data"]["processed_dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(root_dir / config["data"]["clean_path"])
    
    df_filtered.write.mode("overwrite").option("compression", "snappy").parquet(out_path)
    print(f"Wrote clean data to {out_path}")
    spark.stop()

if __name__ == "__main__":
    run_cleaning()
