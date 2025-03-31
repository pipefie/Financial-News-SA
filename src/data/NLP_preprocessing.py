# nlp_preprocessing.py
"""
Module for Spark NLP-based text preprocessing intended for
financial news analysis, suitable for feeding into DistilRoBERTa later.

Example usage in another script or Jupyter notebook:
-----------------------------------------------------
from nlp_preprocessing import PreprocessingPipeline

# Initialize pipeline (needs SparkSession)
preproc = PreprocessingPipeline(spark)

df_raw = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)
df_clean = preproc.run(df_raw, text_col="Article")

# df_clean now has a column 'final_text' which is a single string
# suitable for DistilRoBERTa or other huggingface transformer usage.
-----------------------------------------------------
"""

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (
    Tokenizer,
    Normalizer,
    StopWordsCleaner,
    LemmatizerModel
)
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.sql.functions import concat_ws, col


class PreprocessingPipeline:
    def __init__(self, spark_session):
        """
        Initialize the Spark NLP pipeline components.
        :param spark_session: an active SparkSession
        """
        self.spark = spark_session

        # Document Assembler
        self.document_assembler = (
            DocumentAssembler()
            .setInputCol("raw_text")
            .setOutputCol("document")
        )

        # Tokenizer
        self.tokenizer = (
            Tokenizer()
            .setInputCols(["document"])
            .setOutputCol("token")
        )

        # Normalizer (remove punctuation, keep letters/numbers, lowercase)
        self.normalizer = (
            Normalizer()
            .setInputCols(["token"])
            .setOutputCol("normalized")
            .setLowercase(True)
            .setCleanupPatterns(["""[^\\p{L}\\p{N}]+"""])  # keep letters & digits
        )

        # StopWords Cleaner
        self.stopwords_cleaner = (
            StopWordsCleaner()
            .setInputCols(["normalized"])
            .setOutputCol("cleanTokens")
            .setCaseSensitive(False)
        )

        # Lemmatizer Model (English)
        # If not downloaded, call LemmatizerModel.pretrained("lemma_antbnc", "en") 
        # or replace with the model you prefer
        self.lemmatizer = (
            LemmatizerModel.pretrained("lemma_antbnc", "en")
            .setInputCols(["cleanTokens"])
            .setOutputCol("lemma")
        )

        # Finisher to convert Spark NLP annotations back to array<string>
        self.finisher = (
            Finisher()
            .setInputCols(["lemma"])
            .setOutputCols(["finished_tokens"])
            .setIncludeKeys(False)
        )

        # Build the pipeline
        self.pipeline = Pipeline(
            stages=[
                self.document_assembler,
                self.tokenizer,
                self.normalizer,
                self.stopwords_cleaner,
                self.lemmatizer,
                self.finisher
            ]
        )

    def run(self, df: DataFrame, text_col: str = "text") -> DataFrame:
        """
        Run the Spark NLP pipeline on a DataFrame containing text.
        :param df: Spark DataFrame with text_col as a column of text strings
        :param text_col: name of the column holding raw text (default 'text')
        :return: DataFrame with an extra column 'final_text' for DistilRoBERTa
        """
        # 1. Rename the user-specified text_col to 'raw_text' for pipeline input
        #    (Spark NLP pipeline references 'raw_text')
        df_pre = df.withColumnRenamed(text_col, "raw_text")

        # 2. Fit+transform with pipeline
        model = self.pipeline.fit(df_pre)
        df_transformed = model.transform(df_pre)

        # 3. Create a single string column from the array of tokens
        #    for DistilRoBERTa (which expects plain text).
        df_result = df_transformed.withColumn(
            "final_text",
            concat_ws(" ", col("finished_tokens"))
        )

        # 4. Optionally drop intermediate columns if you don't need them
        #    e.g. 'raw_text', 'document', 'token', etc.
        #    We'll keep them for debugging, but you can remove them:
        # df_result = df_result.drop("raw_text", "document", "token", "normalized",
        #                            "cleanTokens", "lemma", "finished_tokens")

        return df_result
