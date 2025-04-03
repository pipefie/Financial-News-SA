# %% [markdown]
# **Jupyter notebook for test of any type**

# %%
import sys
import os
import pandas as pd
from datasets import load_dataset
import os
import numpy as np
import pyspark as ps
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
import langdetect 
from pyspark.sql.types import StringType, BooleanType
import matplotlib.pyplot as plt
import json
import sparknlp

# Get the absolute path of the project root
project_root = os.path.abspath("..")  # Adjust if necessary

# Add the src directory to Python's path
sys.path.append(os.path.join(project_root, "src"))

if project_root not in sys.path:
    sys.path.append(project_root)


# %%
from data import s3_utils as s3u
from data import NLP_preprocessing as prep

# %%
bucket= "financialdata-sa"
news_external = "RawNews/All_external.csv"

# %%
news1 = s3u.get_csv_as_spark(bucket_name=bucket, s3_key=news_external)

# %%
news1.show(5)

# %%
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Data Exploration") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# %%
spark = SparkSession.builder \
    .appName("Spark NLP") \
    .master("local[*]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()

# %%
spark = SparkSession.builder \
    .appName("Spark NLP") \
    .master("local[*]") \
    .config("spark.driver.memory", "20g") \
    .config("spark.executor.memory", "20g") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3") \
    .getOrCreate()

# %%
# other option for spark session: spark = sparknlp.start()

# %%
df_csv = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("../data/raw/nasdaq_exteral_data.csv")

# %%
df_csv.printSchema()

# %%
df_csv.show(50)

# %%

df_csv_5gb = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("escape", "\"") \
    .option("quote", "\"") \
    .option("delimiter", ",") \
    .option("mode", "PERMISSIVE") \
    .load("../data/raw/All_external.csv")

# %%
df_csv_5gb.describe()

# %%
df_csv_5gb.show(200)

# %%
df_csv_5gb.sort(asc("Date")).show(60)

# %%
df_csv_5gb.printSchema()

# %%
df_csv_5gb.select(min("Date").alias("min_date"), max("Date").alias("max_date")).show()

# %%
df_csv_5gb.groupBy("Stock_symbol").agg(count("*").alias("ticker_count")) \
       .orderBy("Stock_symbol", ascending=True) \
       .show(50)  # Show top 50 tickers

# %%
df_csv_5gb.groupby(date_trunc("month", "date").alias("month")) \
       .count() \
       .orderBy("month") \
       .show()

# %%
df_csv_5gb.show(10)

# %% [markdown]
# cheaper heuristic to validate the language of the news 

# %%
import string

def is_likely_english(text, threshold=0.75):
    if text is None:
        # Here we "pass" the row through
        return True
    if not text:
        return False
    # Keep only letters for ratio calc, or all chars.
    total_chars = len(text)
    if total_chars == 0:
        return False
    
    # Count how many are "basic ASCII" or in the [A-Za-z] range
    ascii_letters = sum(ch in string.printable for ch in text)
    
    ratio = ascii_letters / total_chars
    return ratio >= threshold

# %%
is_english_udf = udf(lambda text: is_likely_english(text, 0.9), BooleanType())

df_filtered = (
  df_csv_5gb
    .withColumn("likely_en_title", is_english_udf(col("Article_title")))
    .withColumn("likely_en_article", is_english_udf(col("Article")))
    # Keep the row if either condition is True
    .filter("likely_en_title = true AND likely_en_article = true")
)

# %%
df_filtered.write.mode("overwrite").parquet("../data/raw/english_only.parquet")

# %%
df_english_only = spark.read.parquet("../data/raw/english_only.parquet")

# %%
df_english_only.sort(asc("Date")).show(60)

# %%
df_english_only.select(min("Date").alias("min_date"), max("Date").alias("max_date")).show()

# %%
df_final = df_english_only.drop("Date_parsed").drop("likely_en_title").drop("likely_en_article")

# %%
df_final.show(10)

# %%
df_final.groupBy("Stock_symbol").agg(count("*").alias("ticker_count")) \
       .orderBy("Stock_symbol", ascending=True) \
       .show(50)  # Show top 50 tickers

# %%
df_time = df_final.withColumn("date_day", to_date("Date", "yyyy-MM-dd"))


# %%
df_time_count = df_time.groupBy("date_day").agg(count("*").alias("articles_count")) \
                      .orderBy("date_day")

# %%
pdf_time = df_time_count.toPandas()

# %% [markdown]
# # Distribution of articles over time 

# %%
plt.plot(pdf_time["date_day"], pdf_time["articles_count"])
plt.xlabel("Date")
plt.ylabel("Number of articles")
plt.title("Daily Article Volume Over Time")
plt.show()

# %% [markdown]
# # Distribution of tickers 

# %%
df_ticker_count = df_final.groupBy("Stock_symbol").agg(count("*").alias("ticker_count")) \
                   .orderBy("ticker_count", ascending=False)

# %%
df_ticker_count_notnull = df_ticker_count.filter("Stock_symbol IS NOT NULL")

# %%
pdf_ticker_notnull = df_ticker_count_notnull.limit(20).toPandas()

# %%
plt.bar(pdf_ticker_notnull["Stock_symbol"], pdf_ticker_notnull["ticker_count"])
plt.xticks(rotation=90)
plt.ylabel("Number of articles")
plt.title("Top 20 Tickers by Article Count")
plt.show()

# %%
df_ticker_count_asc = df_final.groupBy("Stock_symbol").agg(count("*").alias("ticker_count")) \
                   .orderBy("ticker_count", ascending=True)

# %%
df_ticker_count_notnull_asc = df_ticker_count_asc.filter("Stock_symbol IS NOT NULL")

# %%
pdf_ticker_notnull_asc = df_ticker_count_notnull_asc.limit(20).toPandas()

# %%
plt.bar(pdf_ticker_notnull_asc["Stock_symbol"], pdf_ticker_notnull_asc["ticker_count"])
plt.xticks(rotation=90)
plt.ylabel("Number of articles")
plt.title("Bottom 20 Tickers by Article Count")
plt.show()

# %%
df_len = df_final.withColumn("article_len", length("Article"))
# For a quick stats
df_len.describe(["article_len"]).show()

# %%
pdf_len = df_len.select("article_len").sample(fraction=0.1, seed=42).toPandas()  

# %%
# sample 10% 
plt.hist(pdf_len["article_len"], bins=100)
plt.xlabel("Article Length")
plt.ylabel("Frequency")
plt.xlim(0, 10000)
plt.title("Distribution of Article Lengths (sample)")
plt.show()

# %% [markdown]
# # Quick word frecuency

# %%
df_words = df_final.withColumn("title_lower", lower("Article_title")) \
             .withColumn("title_tokens", split("title_lower", "\\s+")) \
             .select(explode("title_tokens").alias("token"))

df_token_count = df_words.groupBy("token").count().orderBy("count", ascending=False)
df_token_count.show(50)

# %% [markdown]
# # Date vs Ticker Heatmap

# %%
df_ticker_day = df_time.groupBy("date_day", "Stock_symbol").count()

# %%
pdf_ticker_day = df_ticker_day.toPandas()

# %%
pivoted = pdf_ticker_day.pivot(index="date_day", columns="Stock_symbol", values="count").fillna(0)

# %%
plt.imshow(pivoted.values, aspect='auto')
plt.xlabel("Ticker")
plt.ylabel("Date")
plt.title("News Coverage Heatmap")
plt.colorbar()
plt.show()

# %% [markdown]
# top 20 tickers by total coverage

# %%
top_tickers = (
    df_time.groupBy("Stock_symbol")
      .agg(count("*").alias("cnt"))
      .orderBy("cnt", ascending=False)
      .limit(20)
)

# %%
top_ticker_list = [row["Stock_symbol"] for row in top_tickers.collect()]

# %%
df_sub = df_time.filter(df_time["Stock_symbol"].isin(top_ticker_list))

# %%
df_ticker_day_top20 = df_sub.groupBy("date_day", "Stock_symbol").count()

# %%
pdf_ticker_day_top = df_ticker_day_top20.toPandas()

# %%
pivoted_top = pdf_ticker_day_top.pivot(index="date_day", columns="Stock_symbol", values="count").fillna(0)

# %%
plt.imshow(pivoted_top.values, aspect='auto')
plt.xlabel("Ticker")
plt.ylabel("Date")
plt.title("News Coverage Heatmap")
plt.colorbar()
plt.show()

# %%
pivoted_top = pdf_ticker_day_top.pivot(index="date_day", columns="Stock_symbol", values="count").fillna(0)

# %%
df_english_only.count()

# %%
df_count = df_time.groupBy("Stock_symbol", "date_day").agg(count("*").alias("news_count"))

# %%
df_count = df_count.orderBy(["Stock_symbol", "date_day"])

# %%
pdf_count = df_count.toPandas()

# %%
pivoted = pdf_count.pivot(index="date_day", columns="Stock_symbol", values="news_count").fillna(0)

# %%
# Example: top 10 tickers by total coverage
ticker_agg = df_count.groupBy("Stock_symbol").sum("news_count").orderBy("sum(news_count)", ascending=False)
top_tickers = [row["Stock_symbol"] for row in ticker_agg.limit(10).collect()]

df_top = df_count.filter(df_count["Stock_symbol"].isin(top_tickers))
pdf_top = df_top.toPandas()

# %%
df_top

# %%
pivoted_top = pdf_top.pivot(index="date_day", columns="Stock_symbol", values="news_count").fillna(0)

# %%
pivoted_top

# %%
for ticker in pivoted_top:
    ticker_data = pdf_count[pdf_count["Stock_symbol"] == ticker]
    # Sort by date
    ticker_data = ticker_data.sort_values("date_day")
    
    plt.figure()  # new figure
    plt.plot(ticker_data["date_day"], ticker_data["news_count"])
    plt.title(f"Daily News Count for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# More heavier but accurate function based language detection 

# %%
def detect_lang(text):
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

detect_lang_udf = udf(detect_lang, StringType())

df_lang = df_csv_5gb.withColumn("detected_lang", detect_lang_udf(df_csv_5gb["Article_title"]))

df_english_only = df_lang.filter(df_lang["detected_lang"] == "en")

# %% [markdown]
# ## Stock symbol recognition for data enhancement

# %%
df_final.show(10)

# %% [markdown]
# Data Scraped from: https://stock-screener.org/stock-list.aspx?alpha=A

# %%
with open("../data/raw/ticker_dict.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert lists back to sets
ticker_dict = {k: set(v) for k, v in data.items()}

# %%
df_range = (
    df_final
    .groupBy("Stock_symbol")
    .agg(
        min("Date").alias("min_date"),
        max("Date").alias("max_date")
    )
    .orderBy("Stock_symbol")  # optional for nice ordering
)

df_range.show(50, truncate=False)

# %% [markdown]
# Now the intent is to fill the NULL Stock symbols in the Article title or Article column with a dictionary of synonyms created from the data scraped and some functions for processing

# %%
### Combine text to find the stock in either column if anyone is NULL 

df_combined = df_final.withColumn(
    "combined_text", 
    concat_ws(" ", col("Article"), col("Article_title"))
)

# %%
# UDF
def deduce_ticker_from_text(text):
    if not text:
        return None
    text_lower = text.lower()
    for ticker, synonyms in broadcast_dict.value.items():
        for syn in synonyms:
            if syn in text_lower:
                return ticker
    return None

# %%
deduce_ticker_udf = udf(deduce_ticker_from_text, StringType())

# %%
df_deduced = df_combined.withColumn(
    "deduced_ticker",
    deduce_ticker_udf(col("combined_text"))
)

# %%
df_deduced.sort(desc("Date")).show(10)

# %% [markdown]
# ### NLP based deduction

# %%
rows = []
for ticker, synonyms in ticker_dict.items():
    for syn in synonyms:
        rows.append(Row(synonym=syn, ticker=ticker))

synonyms_df = spark.createDataFrame(rows)  # columns: synonym, ticker

# %%
broadcast_dict = broadcast(synonyms_df)

# %%
synonyms_df.show(10)

# %%
preproc = prep.PreprocessingPipeline(spark)

# %%
df_nlp = preproc.run(df_combined, text_col="combined_text")

# %%
df_nlp = df_nlp.persist()

# %%
df_nlp.select("finished_tokens").show(truncate=False, n=3)

# %%
df_nlp.show(10)

# %%
df_nlp.printSchema()

# %%
# Add a row_id to df_nlp for rejoin after exploding
df_with_id = df_nlp.withColumn("row_id", monotonically_increasing_id())


# %%
#create memory issues because explode makes the df much bigger (memory bottlenecks)

df_exploded = df_with_id.select(
    "row_id", "Stock_symbol",
    explode(col("finished_tokens")).alias("token")
)
joined = df_exploded.join(
    broadcast_dict,
    df_exploded["token"] == synonyms_df["synonym"],
    how="left"
)

# %%
joined.printSchema()

# %%
joined.show(5)

# %%
#Join without explode, using array_contains
df_with_match = df_with_id.crossJoin(synonyms_df) \
    .where(array_contains(col("finished_tokens"), col("synonym")))

# %%
df_matched = df_with_match.groupBy("row_id").agg(
    first("Stock_symbol").alias("original_symbol"),
    collect_set("ticker").alias("matched_tickers")
)

# %%
df_matched = df_matched.persist()

# %%
df_filled = df_matched.withColumn(
    "final_ticker",
    when(
        (col("original_symbol").isNull()) & (size(col("matched_tickers")) > 0),
        expr("matched_tickers[0]")  # pick first matched ticker
    ).otherwise(col("original_symbol"))
)

# %%
df_merged = df_with_id.join(df_filled.select("row_id", "final_ticker"), "row_id", "left") \
                      .drop("row_id")

# %%
df_merged = df_merged.persist()

# %%
df_merged.count()

# %%
df_merged.write.partitionBy("final_ticker").parquet("../data/raw/processed_news.parquet")


