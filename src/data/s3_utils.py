import boto3
import botocore
import os
import pyspark as spark
from pyspark.sql import SparkSession
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1") 

def get_s3_resource(region_name=" eu-north-1"):
    """
    Get an S3 resource using boto3.

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
    

def get_spark_session():
    """Initialize a Spark session with S3A support."""
    jars = [
        "/home/pipe/spark_jars/hadoop-aws-3.3.4.jar",
        "/home/pipe/spark_jars/aws-java-sdk-bundle-1.11.1026.jar",
        "/home/pipe/spark_jars/hadoop-common-3.3.4.jar",
        "/home/pipe/spark_jars/hadoop-auth-3.3.4.jar"
    ]
    
    spark = SparkSession.builder \
        .appName("S3 Read Example") \
        .config("spark.jars", ",".join(jars)) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.fast.upload", "true") \
        .getOrCreate()

    return spark




def list_buckets():
    """
    List all S3 buckets accessible with your AWS credentials.

    Returns:
        list: A list of bucket names.
    """
    s3 = get_s3_resource()
    return [bucket.name for bucket in s3.buckets.all()]


def list_files(bucket_name, prefix=""):
    """
    List all files (objects) in a specified bucket and optional prefix (folder).

    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix (folder path) to filter objects.

    Returns:
        list: A list of object keys (file paths) in the bucket.
    """
    s3 = get_s3_resource()
    bucket = s3.Bucket(bucket_name)
    files = [obj.key for obj in bucket.objects.filter(Prefix=prefix)]
    return files


def get_file(bucket_name, s3_key):
    """
    Retrieve an object's content from S3 as bytes.
    
    Args:
        bucket_name (str): S3 bucket name.
        s3_key (str): Object key (path).
    
    Returns:
        bytes: Content of the object, or None if error.
    """
    s3 = get_s3_resource()
    try:
        obj = s3.Object(bucket_name, s3_key)
        content = obj.get()['Body'].read()
        return content
    except botocore.exceptions.ClientError as e:
        print(f"Error retrieving file: {e}")
        return None

def upload_file(local_file_path, bucket_name, s3_key):
    """
    Upload a file from the local filesystem to an S3 bucket.

    Args:
        local_file_path (str): Path to the local file.
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): The S3 object key (including any prefix/folder).

    Returns:
        bool: True if upload was successful, False otherwise.
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
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): The S3 object key to download.
        local_file_path (str): The destination path on the local filesystem.

    Returns:
        bool: True if download was successful, False otherwise.
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
    Delete a file from an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): The S3 object key to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
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


def create_folder(bucket_name, folder_name):
    """
    "Create" a folder in an S3 bucket. In S3, folders are represented as
    prefixes ending with a slash.

    Args:
        bucket_name (str): Name of the S3 bucket.
        folder_name (str): Name of the folder to create.

    Returns:
        bool: True if the folder "creation" was successful, False otherwise.
    """
    # Ensure the folder name ends with '/'
    if not folder_name.endswith('/'):
        folder_name += '/'
    # Upload an empty object with the folder name as the key.
    return upload_file("", bucket_name, folder_name)


def update_file(local_file_path, bucket_name, s3_key):
    """
    Update a file on S3 by uploading a new version over the existing key.

    Args:
        local_file_path (str): Path to the new local file.
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): The S3 object key to update.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    # In S3, updating is just uploading a file with the same key.
    return upload_file(local_file_path, bucket_name, s3_key)

# -------------------
# Data Loading and Storing with PySpark
# -------------------

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

def store_csv_spark(df, bucket_name, s3_key, mode="overwrite", header=True):
    """
    Write a Spark DataFrame as a CSV file to S3.
    
    Args:
        df (pyspark.sql.DataFrame): DataFrame to store.
        bucket_name (str): S3 bucket name.
        s3_key (str): Destination S3 object key (path).
        mode (str): Write mode (default "overwrite").
        header (bool): Write header row if True.
    
    Returns:
        bool: True if write succeeded, False otherwise.
    """
    s3_uri = f"s3a://{bucket_name}/{s3_key}"
    try:
        df.write.mode(mode).csv(s3_uri, header=header)
        print(f"DataFrame written to {s3_uri}")
        return True
    except Exception as e:
        print(f"Error writing CSV to S3: {e}")
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


# Example usage (you can remove this block when integrating into your project):
if __name__ == "__main__":
    bucket = "financialdata-sa"
    folder = "RawNews/"
    # Test listing buckets
    print("Buckets:", [bucket.name for bucket in get_s3_resource().buckets.all()])
    # Test creating a folder
    create_folder(bucket, folder)
    # Test listing files in the folder
    print("Files in folder:", list_files(bucket, prefix=folder))
