import os
import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType
import torch

# Import necessary Hugging Face tools for model loading and inference
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline



# -------------------------------------------------------------------------------------
# Model Loading and Pipeline Creation
# -------------------------------------------------------------------------------------
_MODEL_CACHE = {}  # Global cache to avoid reloading the model in every UDF call

def load_finetuned_model(model_dir):
    """
    Load the fine-tuned model and tokenizer from the specified directory.
    
    Args:
        model_dir (str): Path to the fine-tuned model directory.
        
    Returns:
        model: The fine-tuned model.
        tokenizer: The corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def get_sentiment_pipeline(model_dir):
    """
    Create and return a sentiment analysis pipeline using the fine-tuned model.
    
    Args:
        model_dir (str): Directory where the fine-tuned model is saved.
        
    Returns:
        pipeline: A Hugging Face sentiment analysis pipeline.
    """
    # Determine device: use GPU if available, otherwise CPU.
    device = 0 if torch.cuda.is_available() else -1

    model, tokenizer = load_finetuned_model(model_dir)
    # Set device=-1 to use CPU; modify if you want to use a GPU.
    sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    return sentiment_pipe

# -------------------------------------------------------------------------------------
# Pandas UDF for Model Inference in Spark
# -------------------------------------------------------------------------------------
@pandas_udf(StringType())
def predict_sentiment_udf(texts: pd.Series) -> pd.Series:
    """
    Pandas UDF for batch sentiment inference.
    
    This function is applied on a Spark DataFrame column (as a pandas Series).
    It loads the fine-tuned model (and caches it to avoid reloading within the partition)
    and then runs inference on the input texts using a Hugging Face pipeline.
    
    Args:
        texts (pd.Series): A pandas Series of text strings.
        
    Returns:
        pd.Series: A pandas Series of predicted sentiment labels as strings.
    """
    # Use an environment variable to specify the model directory;
    # default to "./finetuned_distilroberta" if not provided.
    model_dir = os.getenv("FINETUNED_MODEL_DIR", "./finetuned_distilroberta")
    
    # Cache the pipeline so that each executor only loads it once per partition.
    global _MODEL_CACHE
    if model_dir not in _MODEL_CACHE:
        _MODEL_CACHE[model_dir] = get_sentiment_pipeline(model_dir)
    sentiment_pipe = _MODEL_CACHE[model_dir]
    
    # Convert the incoming batch (a pandas Series) to a list for the pipeline.
    texts_list = texts.tolist()
    
    try:
        # Run the inference pipeline on the entire batch.
        # The pipeline returns a list of dictionaries with keys "label" and "score".
        results = sentiment_pipe(texts_list)
    except Exception as e:
        # If something goes wrong, print error and fill the batch with an error value.
        print("Inference error:", e)
        return pd.Series(["error"] * len(texts_list))
    
    # Extract the predicted label from each result.
    predicted_labels = [result.get("label", "unknown") for result in results]
    return pd.Series(predicted_labels)

# -------------------------------------------------------------------------------------
# Function to Apply Inference on a Spark DataFrame
# -------------------------------------------------------------------------------------
def apply_inference_to_dataframe(spark_df, text_column="final_text"):
    """
    Apply the sentiment prediction Pandas UDF to a Spark DataFrame.
    
    This function takes a Spark DataFrame containing a text column and returns a new
    DataFrame with an added column "predicted_sentiment" containing the model's predictions.
    
    Args:
        spark_df (DataFrame): The input Spark DataFrame.
        text_column (str): The name of the column in spark_df with text to process.
        
    Returns:
        DataFrame: The Spark DataFrame with an additional "predicted_sentiment" column.
    """
    return spark_df.withColumn("predicted_sentiment", predict_sentiment_udf(col(text_column)))
