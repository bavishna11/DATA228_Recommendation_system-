"""Exploratory Data Analysis Module.
Replaces Notebook 1 EDA sections.
"""
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc

def init_spark():
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"]
        ).decode().strip()
    except Exception:
        pass
    spark = SparkSession.builder \
        .appName("Amazon_EDA") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()
    return spark

from src.utils.config_loader import load_config

def run_eda():
    print("Running Exploratory Data Analysis...")
    config, root_dir = load_config()
    spark = init_spark()
    parquet_path = str(root_dir / config["data"]["raw_path"])
    abs_path = "file:///" + os.path.abspath(parquet_path).replace("\\", "/")
    
    try:
        df = spark.read.parquet(abs_path)
    except Exception as e:
        print(f"Could not load data for EDA: {e}")
        return

    print(f"Total records loaded: {df.count():,}")
    df.printSchema()

    # 1. Rating Distribution
    print("--- Rating Distribution ---")
    df.groupBy("rating").count().orderBy("rating").show()

    # 2. Top Products
    print("--- Top 20 Most Reviewed Products ---")
    df.groupBy("parent_asin").count().orderBy(desc("count")).limit(20).show()

    # 3. Power Users
    power_users = df.groupBy("user_id").count().filter(col("count") > 50)
    print("Total Power Users (>50 reviews):", power_users.count())

    spark.stop()

if __name__ == "__main__":
    run_eda()
