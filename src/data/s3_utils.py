import os
import boto3
import botocore
from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --------------------------------------------------------------------
# Load environment variables from .env (for AWS credentials, etc.)
# --------------------------------------------------------------------
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

# --------------------------------------------------------------------
# Boto3 Helpers
# --------------------------------------------------------------------

def get_s3_resource(region_name=AWS_REGION):
    """
    Get an S3 resource using boto3 environment-based credentials.

    Args:
        region_name (str): The AWS region name (default is " eu-north-1").

    Returns:
        boto3.resource: S3 resource instance.
    """
    return boto3.resource(
        "s3", 
        region_name=region_name,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,)

def get_s3_client(region_name=AWS_REGION):
    """Get an S3 client using boto3 with loaded environment credentials."""
    return boto3.client(
        "s3",
        region_name=region_name,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    
# --------------------------------------------------------------------
# Spark Session Initialization
# --------------------------------------------------------------------
def get_spark_session():
    """
    Initialize a Spark session with recommended S3 configurations.
    Includes jars for s3a if needed. Adjust paths and config as needed.
    """
    jars = [
        "/home/pipe/spark_jars/hadoop-aws-3.3.4.jar",
        "/home/pipe/spark_jars/aws-java-sdk-bundle-1.11.1026.jar",
        "/home/pipe/spark_jars/hadoop-common-3.3.4.jar",
        "/home/pipe/spark_jars/hadoop-auth-3.3.4.jar"
    ]
    
    spark = SparkSession.builder \
        .appName("Financial SA") \
        .config("spark.jars", ",".join(jars)) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID) \
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.fast.upload", "true") \
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")\
        .getOrCreate()

    return spark


# --------------------------------------------------------------------
# Basic CRUD and Listing with Caching
# --------------------------------------------------------------------
_LIST_FILES_CACHE = {}


def list_buckets():
    """
    List all S3 buckets accessible with your AWS credentials.

    Returns:
        list: A list of bucket names.
    """
    s3 = get_s3_resource()
    return [bucket.name for bucket in s3.buckets.all()]


def list_files(bucket_name, prefix="", use_cache=True):
    """
    List files (objects) under a prefix in an S3 bucket.
    Includes an in-memory cache to reduce repeated LIST calls.
    """
    cache_key = f"{bucket_name}:{prefix}"
    if use_cache and cache_key in _LIST_FILES_CACHE:
        return _LIST_FILES_CACHE[cache_key]

    s3 = get_s3_resource()
    bucket = s3.Bucket(bucket_name)
    file_keys = [obj.key for obj in bucket.objects.filter(Prefix=prefix)]

    if use_cache:
        _LIST_FILES_CACHE[cache_key] = file_keys

    return file_keys


def create_folder(bucket_name, folder_name):
    """
    "Create" a folder in S3 by ensuring the prefix ends with a slash.
    (S3 folders are virtual—this creates an empty object with that key.)
    
    Args:
        bucket_name (str): The S3 bucket name.
        folder_name (str): The folder name to create.
    
    Returns:
        bool: True if creation (upload of empty object) succeeded.
    """
    if not folder_name.endswith("/"):
        folder_name += "/"
    return upload_file("", bucket_name, folder_name)

def upload_file(local_file_path, bucket_name, s3_key):
    """
    Upload a file from the local filesystem to an S3 bucket.
    For very large files, consider using multipart upload (handled automatically by boto3).
    
    Args:
        local_file_path (str): Local file path.
        bucket_name (str): S3 bucket name.
        s3_key (str): Destination S3 key.
    
    Returns:
        bool: True if upload succeeded, False otherwise.
    """
    s3 = get_s3_resource()
    try:
        s3.Bucket(bucket_name).upload_file(local_file_path, s3_key)
        print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
        return True
    except botocore.exceptions.ClientError as e:
        print(f"Error uploading file: {e}")
        return False

def download_file(bucket_name, s3_key, local_file_path):
    """
    Download a file from an S3 bucket to a local path.
    
    Args:
        bucket_name (str): S3 bucket name.
        s3_key (str): S3 object key.
        local_file_path (str): Destination local file path.
    
    Returns:
        bool: True if download succeeded, False otherwise.
    """
    s3 = get_s3_resource()
    try:
        s3.Bucket(bucket_name).download_file(s3_key, local_file_path)
        print(f"Downloaded s3://{bucket_name}/{s3_key} to {local_file_path}")
        return True
    except botocore.exceptions.ClientError as e:
        print(f"Error downloading file: {e}")
        return False

def delete_file(bucket_name, s3_key):
    """
    Download a file from an S3 bucket to a local path.
    
    Args:
        bucket_name (str): S3 bucket name.
        s3_key (str): S3 object key.
        local_file_path (str): Destination local file path.
    
    Returns:
        bool: True if download succeeded, False otherwise.
    """
    s3 = get_s3_resource()
    try:
        obj = s3.Object(bucket_name, s3_key)
        obj.delete()
        print(f"Deleted s3://{bucket_name}/{s3_key}")
        return True
    except botocore.exceptions.ClientError as e:
        print(f"Error deleting file: {e}")
        return False

def update_file(local_file_path, bucket_name, s3_key):
    """
    Update a file on S3 by uploading a new version over the existing key.
    
    Args:
        local_file_path (str): Local file path.
        bucket_name (str): S3 bucket name.
        s3_key (str): S3 object key to update.
    
    Returns:
        bool: True if update succeeded, False otherwise.
    """
    return upload_file(local_file_path, bucket_name, s3_key)

# --------------------------------------------------------------------
# AWS CLI Sync Functions for Large Bulk Transfers
# --------------------------------------------------------------------
def s3_sync_upload(local_dir, bucket_name, s3_prefix):
    """
    Use AWS CLI's s3 sync command to upload an entire local directory to S3.
    This is more efficient for very large data transfers.
    
    Args:
        local_dir (str): Local directory path to sync.
        bucket_name (str): S3 bucket name.
        s3_prefix (str): Destination S3 prefix (folder path).
    
    Returns:
        bool: True if sync command succeeded, False otherwise.
    """
    s3_uri = f"s3://{bucket_name}/{s3_prefix}"
    try:
        result = subprocess.run(["aws", "s3", "sync", local_dir, s3_uri],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Sync upload successful: {result.stdout.decode()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during s3 sync upload: {e.stderr.decode()}")
        return False

def s3_sync_download(bucket_name, s3_prefix, local_dir):
    """
    Use AWS CLI's s3 sync command to download data from S3 to a local directory.
    
    Args:
        bucket_name (str): S3 bucket name.
        s3_prefix (str): S3 prefix (folder path) to sync.
        local_dir (str): Local directory to download files into.
    
    Returns:
        bool: True if sync command succeeded, False otherwise.
    """
    s3_uri = f"s3://{bucket_name}/{s3_prefix}"
    try:
        result = subprocess.run(["aws", "s3", "sync", s3_uri, local_dir],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Sync download successful: {result.stdout.decode()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during s3 sync download: {e.stderr.decode()}")
        return False

# --------------------------------------------------------------------
# S3 Select for Large CSVs
# --------------------------------------------------------------------

def s3_select_csv(
    bucket_name,
    s3_key,
    sql_expression="SELECT * FROM RawNews/All_external.csv",
    input_serialization=None,
    output_serialization=None,
):
    """
    Use S3 Select to query a large CSV directly in S3, reducing data transfer.

    Args:
        bucket_name (str): S3 bucket name.
        s3_key (str): Key of the CSV file.
        sql_expression (str): SQL expression for S3 Select (default: SELECT all from All_external.csv).
        input_serialization (dict): Format info for the CSV (delimiter, header, etc.).
        output_serialization (dict): Format info for the output.
    
    Returns:
        bytes: The raw CSV data matching the query, or None if error.
    
    Example usage:
        result = s3_select_csv(
            "my-bucket",
            "path/to/large.csv",
            sql_expression="SELECT s.* FROM s3object s WHERE s.col1 = 'value'"
        )
    """
    if input_serialization is None:
        # Default to CSV with header
        input_serialization = {
            "CSV": {"FileHeaderInfo": "USE"},
            "CompressionType": "NONE",
        }
    if output_serialization is None:
        output_serialization = {"CSV": {}}

    s3_client = get_s3_client()
    try:
        response = s3_client.select_object_content(
            Bucket=bucket_name,
            Key=s3_key,
            ExpressionType="SQL",
            Expression=sql_expression,
            InputSerialization=input_serialization,
            OutputSerialization=output_serialization,
        )

        # The response is a streaming event. We need to concatenate them.
        result_data = []
        for event in response["Payload"]:
            if "Records" in event:
                result_data.append(event["Records"]["Payload"])
        return b"".join(result_data)

    except botocore.exceptions.ClientError as e:
        print(f"S3 Select error: {e}")
        return None

# --------------------------------------------------------------------
# Spark CSV / Parquet Integration
# --------------------------------------------------------------------

def get_csv_as_spark(bucket_name, s3_key, spark=None, header=True, inferSchema=True):
    """
    Load a CSV file from S3 directly into a Spark DataFrame.
    
    Args:
        bucket_name (str): S3 bucket name.
        s3_key (str): S3 object key (path to CSV file).
        spark (SparkSession, optional): Active Spark session. If None, creates one.
        header (bool): True if the CSV file has a header row.
        inferSchema (bool): Whether to infer the schema automatically.
    
    Returns:
        pyspark.sql.DataFrame: The loaded Spark DataFrame.
    """
    if spark is None:
        spark = get_spark_session()
    
    # Construct S3 URI using the s3a protocol
    s3_uri = f"s3a://{bucket_name}/{s3_key}"
    try:
        df = spark.read.csv(s3_uri, header=header, inferSchema=inferSchema)
        return df
    except Exception as e:
        print(f"Error reading CSV from S3: {e}")
        return None
    
def load_partitioned_csv_as_spark(bucket_name, prefix, spark=None, header=True, inferSchema=True):
    """
    Load multiple CSV files from S3 that share a common prefix (i.e. partitioned data)
    into a single Spark DataFrame. This supports parallel reads over many files.
    
    Args:
        bucket_name (str): S3 bucket name.
        prefix (str): Common prefix (folder) for the CSV files.
        spark (SparkSession, optional): Existing Spark session, or create one.
        header (bool): CSV header presence.
        inferSchema (bool): Whether to infer schema.
    
    Returns:
        pyspark.sql.DataFrame: Combined DataFrame from all matching CSV files.
    """
    if spark is None:
        spark = get_spark_session()
    # Ensure prefix ends with "/" then append wildcard
    if not prefix.endswith("/"):
        prefix += "/"
    s3_uri = f"s3a://{bucket_name}/{prefix}*.csv"
    try:
        df = spark.read.csv(s3_uri, header=header, inferSchema=inferSchema)
        return df
    except Exception as e:
        print(f"Error reading partitioned CSV from S3: {e}")
        return None

def store_csv_spark(df, bucket_name, s3_key, mode="overwrite", header=True, partitions=1):
    """
    Write a Spark DataFrame as a CSV file to S3.
    
    Args:
        df (pyspark.sql.DataFrame): DataFrame to store.
        bucket_name (str): S3 bucket name.
        s3_key (str): Destination S3 object key (path).
        mode (str): Write mode (default "overwrite").
        header (bool): Write header row if True.
        partitions: Number of partitions (output files)
    
    Returns:
        bool: True if write succeeded, False otherwise.
    """
    try:
        df = df.coalesce(partitions)  # reduce the number of output files
        s3_uri = f"s3a://{bucket_name}/{s3_key}"
        df.write.mode(mode).csv(s3_uri, header=header)
        print(f"DataFrame written to {s3_uri} with {partitions} part-file(s).")
        return True
    except Exception as e:
        print(f"Error writing CSV to S3: {e}")
        return False

def store_parquet_spark(df, bucket_name, prefix, mode="overwrite", partitions=1):
    """
    Write a Spark DataFrame to S3 as Parquet, which is more efficient than CSV for large data.

    Args:
        df (pyspark.sql.DataFrame): DataFrame to store.
        bucket_name (str): S3 bucket name.
        prefix (str): Destination S3 object key (path).
        mode (str): Write mode (default "overwrite").
        partitions: Number of partitions (output files)
    
    Returns:
        bool: True if write succeeded, False otherwise.
    """
    try:
        df = df.coalesce(partitions)
        s3_uri = f"s3a://{bucket_name}/{prefix}"
        df.write.mode(mode).parquet(s3_uri)
        print(f"DataFrame written to {s3_uri} (Parquet) with {partitions} part-file(s).")
        return True
    except Exception as e:
        print(f"Error writing Parquet to S3: {e}")
        return False

def get_csv_as_pandas(bucket_name, s3_key, spark=None, header=True, inferSchema=True):
    """
    Load a CSV file from S3 into a Spark DataFrame, then convert it to a pandas DataFrame.
    This is useful for small-to-moderate datasets or for prototyping.
    
    Args:
        bucket_name (str): S3 bucket name.
        s3_key (str): S3 object key (path to CSV file).
        spark (SparkSession, optional): Active Spark session.
        header (bool): True if CSV has header.
        inferSchema (bool): Whether to infer schema.
    
    Returns:
        pandas.DataFrame: The loaded pandas DataFrame, or None on error.
    """
    df_spark = get_csv_as_spark(bucket_name, s3_key, spark, header, inferSchema)
    if df_spark is not None:
        try:
            return df_spark.toPandas()
        except Exception as e:
            print(f"Error converting Spark DataFrame to pandas: {e}")
            return None
    return None

def repartition_df(df, num_partitions):
    """
    Repartition a Spark DataFrame to a specified number of partitions.
    
    Args:
        df (pyspark.sql.DataFrame): The DataFrame to repartition.
        num_partitions (int): Desired number of partitions.
    
    Returns:
        pyspark.sql.DataFrame: The repartitioned DataFrame.
    """
    return df.repartition(num_partitions)

def cache_dataframe(df):
    """
    Cache a Spark DataFrame in memory to avoid re-reading from S3.
    
    Args:
        df (pyspark.sql.DataFrame): The DataFrame to cache.
    
    Returns:
        pyspark.sql.DataFrame: The cached DataFrame.
    """
    return df.cache()

# --------------------------------------------------------------------
# Explanation: What Does `b"".join(result_data)` Do?
# --------------------------------------------------------------------
# In the s3_select_csv function, S3 Select returns its results as a stream of events.
# Each event may contain a bytes object (event["Records"]["Payload"]) that is a fragment of the full result.
# The expression:
#
#     return b"".join(result_data)
#
# takes a list of these byte fragments (result_data) and concatenates them into a single bytes object.
# This single bytes object contains the complete data returned by the S3 Select query.
#
# --------------------------------------------------------------------
# Example Usage (for testing)
# --------------------------------------------------------------------

# Example usage:
if __name__ == "__main__":
    bucket = "financialdata-sa"
    folder = "RawNews/"

    # List buckets
    print("Buckets:", list_buckets())

    # List files in folder
    files = list_files(bucket, prefix=folder)
    print("Files in folder:", files)

    # Example: S3 Select usage (simple case)
    # data = s3_select_csv(
    #     bucket,
    #     "big_data.csv",
    #     sql_expression="SELECT s.* FROM s3object s WHERE s.columnX = 'value'"
    # )
    # if data:
    #     print("First 500 bytes of query result:", data[:500])
    
    # Spark read CSV example
    # spark_session = get_spark_session()
    # df_spark = get_csv_as_spark(bucket, folder + "some_large_file.csv", spark_session)
    # if df_spark:
    #     store_parquet_spark(df_spark, bucket, folder + "parquet_output", partitions=2)