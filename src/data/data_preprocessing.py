#!/usr/bin/env python3
"""
Preprocess financial news and historical price data using PySpark.
Assumptions:
- News data is stored in a CSV file with at least a 'news_text' column.
- Price data is stored in a CSV file with at least a 'date' column and price information.
- Data paths can be adjusted to point to your local mount or cloud-synced directories.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, to_date, trim
import os

def create_spark_session(app_name="FinancialDataPreprocessing"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_price_data(spark, price_path):
    # Load price data
    df = spark.read.csv(price_path, header=True, inferSchema=True)
    return df

def clean_price_data(df):
    # Convert 'date' column to proper date format and drop rows with missing values
    df_clean = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    # Add any additional cleaning such as handling missing prices or filtering date ranges
    df_clean = df_clean.dropna(subset=["date"])
    return df_clean

def save_cleaned_data(df, output_path):
    # Write the cleaned data to CSV. You can also use Parquet for better performance.
    df.write.mode("overwrite").csv(output_path, header=True)
    print(f"Cleaned data saved to {output_path}")

def run_preprocessing_flow(config):
    """
    Main entry point for data preprocessing using a config dictionary.
    """
    spark = create_spark_session()

    # Local paths from config
    raw_data_path = config["local_paths"]["raw_data"]
    processed_data_path = config["local_paths"]["processed_data"]

    # Example: we assume the price CSVs are already downloaded locally
    # E.g., data/raw/financial_news.csv and data/raw/price_data/AAPL_historical.csv
    price_raw_path = os.path.join(raw_data_path, "price_data", "AAPL_historical.csv")

    price_clean_output = os.path.join(processed_data_path, "clean_price_data")

    # Load data
    price_df = load_price_data(spark, price_raw_path)

    # Clean data
    price_clean_df = clean_price_data(price_df)

    print("Sample cleaned price data:")
    price_clean_df.show(5, truncate=False)

    # Save cleaned data
    save_cleaned_data(price_clean_df, price_clean_output)

    spark.stop()

if __name__ == "__main__":
    default_config = {
        "local_paths": {
            "raw_data": "../../data/raw",
            "processed_data": "../../data/processed"
        }
    }
    run_preprocessing_flow(default_config)
