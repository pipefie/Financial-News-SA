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

def create_spark_session(app_name="FinancialDataPreprocessing"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_news_data(spark, news_path):
    # Load news data; adjust header and schema settings as necessary
    df = spark.read.csv(news_path, header=True, inferSchema=True)
    return df

def load_price_data(spark, price_path):
    # Load price data
    df = spark.read.csv(price_path, header=True, inferSchema=True)
    return df

def clean_news_data(df):
    # Example cleaning: trim whitespace, convert text to lowercase, remove punctuation
    df_clean = df.withColumn("news_text", trim(col("news_text"))) \
                 .withColumn("clean_text", lower(col("news_text"))) \
                 .withColumn("clean_text", regexp_replace(col("clean_text"), "[^a-zA-Z0-9\\s]", ""))
    return df_clean

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

def main():
    spark = create_spark_session()

    # Define paths: update these paths to match your cloud-synced folders or local directories.
    news_raw_path = "../../data/raw/financial_news.csv"       # Change this to your file's location
    price_raw_path = "../../data/raw/price_AAPL.csv"            # Change to your price data file location
    news_clean_output = "../../data/processed/clean_news_data"
    price_clean_output = "../../data/processed/clean_price_data"

    # Load data
    news_df = load_news_data(spark, news_raw_path)
    price_df = load_price_data(spark, price_raw_path)

    # Clean data
    news_clean_df = clean_news_data(news_df)
    price_clean_df = clean_price_data(price_df)

    # Show sample data
    print("Sample cleaned news data:")
    news_clean_df.show(5, truncate=False)
    
    print("Sample cleaned price data:")
    price_clean_df.show(5, truncate=False)

    # Save cleaned data (local storage is recommended during development for speed)
    save_cleaned_data(news_clean_df, news_clean_output)
    save_cleaned_data(price_clean_df, price_clean_output)

    spark.stop()

if __name__ == "__main__":
    main()
