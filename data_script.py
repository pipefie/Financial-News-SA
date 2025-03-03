from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace

# Initialize Spark session
spark = SparkSession.builder.appName("FinancialSentimentAnalysis").getOrCreate()

# Load dataset (CSV, JSON, etc.)
df = spark.read.csv("path/to/financial_news.csv", header=True, inferSchema=True)

# Clean text: lowercasing and removing punctuation
df_clean = df.withColumn("clean_text", lower(col("news_text"))) \
             .withColumn("clean_text", regexp_replace(col("clean_text"), "[^a-zA-Z0-9\\s]", ""))

df_clean.show(5)
