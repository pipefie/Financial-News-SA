from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import json
import tempfile
import pyspark as spark
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_AUTH_URI = os.getenv("GOOGLE_AUTH_URI")
GOOGLE_TOKEN_URI = os.getenv("GOOGLE_TOKEN_URI")
GOOGLE_CERT_URL = os.getenv("GOOGLE_CERT_URL")
GOOGLE_REDIRECT_URIS = os.getenv("GOOGLE_REDIRECT_URIS")

def authenticate_drive():
    """Authenticate with Google Drive using environment variables."""
    gauth = GoogleAuth()
    
    
    # DEBUG: Print environment variables
    print("GOOGLE_CLIENT_ID:", os.getenv("GOOGLE_CLIENT_ID"))
    print("GOOGLE_CLIENT_SECRET:", os.getenv("GOOGLE_CLIENT_SECRET"))
    print("GOOGLE_AUTH_URI:", os.getenv("GOOGLE_AUTH_URI"))
    print("GOOGLE_TOKEN_URI:", os.getenv("GOOGLE_TOKEN_URI"))
    print("GOOGLE_CERT_URL:", os.getenv("GOOGLE_CERT_URL"))
    print("GOOGLE_REDIRECT_URIS:", os.getenv("GOOGLE_REDIRECT_URIS"))

    # Check if any of them are None (unset)
    if None in [
        os.getenv("GOOGLE_CLIENT_ID"),
        os.getenv("GOOGLE_CLIENT_SECRET"),
        os.getenv("GOOGLE_AUTH_URI"),
        os.getenv("GOOGLE_TOKEN_URI"),
        os.getenv("GOOGLE_CERT_URL"),
        os.getenv("GOOGLE_REDIRECT_URIS"),
    ]:
        print("⚠️ Missing required environment variables!")
        return None

    # Ensure GOOGLE_REDIRECT_URIS is properly formatted
    try:
        redirect_uris = json.loads(os.getenv("GOOGLE_REDIRECT_URIS", '["http://localhost"]'))
    except json.JSONDecodeError:
        print("⚠️ Invalid format for GOOGLE_REDIRECT_URIS. It should be a valid JSON string.")
        return None

 # Create temporary JSON file
    client_config = {
        "installed": {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
            "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("GOOGLE_CERT_URL"),
            "redirect_uris": json.loads(os.getenv("GOOGLE_REDIRECT_URIS", '["http://localhost"]'))
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_json:
        json.dump(client_config, temp_json)
        temp_json_path = temp_json.name

    # Load the temporary JSON file
    gauth.LoadClientConfigFile(temp_json_path)

    try:
        gauth.LocalWebserverAuth()  # Authenticate user via browser
    except Exception as e:
        print(f"Authentication failed: {e}")
        return None

    return GoogleDrive(gauth)

def list_drive_files(folder_id):
    drive = authenticate_drive()
    if not drive:
        return None
    
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return [(file['title'], file['id']) for file in file_list]

def download_file_from_drive(file_id, local_path):
    """Download a file from Google Drive to a local path."""
    drive = authenticate_drive()
    if not drive:
        return None
    
    try:
        file_obj = drive.CreateFile({'id': file_id})
        file_obj.GetContentFile(local_path)
        print(f"File with ID {file_id} downloaded to {local_path}.")
        return local_path
    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        return None

def create_folder(folder_name, parent_folder_id=None):
    """Create a new folder in Google Drive."""
    drive = authenticate_drive()
    if not drive:
        return None
    
    folder_metadata = {'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    
    if parent_folder_id:
        folder_metadata['parents'] = [{'id': parent_folder_id}]
        
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    print(f"Folder '{folder_name}' created with ID: {folder['id']}")
    return folder['id']

def create_or_update_file(file_path, file_title, folder_id=None):
    """Upload or update a file in Google Drive."""
    drive = authenticate_drive()
    if not drive:
        return None
    
    # Search for file with the same title in the given folder
    query = f"title='{file_title}' and trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    file_list = drive.ListFile({'q': query}).GetList()

    if file_list:
        # If file exists, update it
        file = file_list[0]
        file.SetContentFile(file_path)
        file.Upload()
        print(f"File '{file_title}' updated.")
    else:
        # Create a new file
        file_metadata = {'title': file_title}
        if folder_id:
            file_metadata['parents'] = [{'id': folder_id}]
        file = drive.CreateFile(file_metadata)
        file.SetContentFile(file_path)
        file.Upload()
        print(f"File '{file_title}' created with ID: {file['id']}")
    return file['id']

def get_file_id_by_title(file_title, folder_id=None):
    drive = authenticate_drive()
    if not drive:
        return None
    query = f"title='{file_title}' and trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    file_list = drive.ListFile({'q': query}).GetList()
    if file_list:
        return file_list[0]['id']
    else:
        return None
    
    
def load_csv_from_drive(file_id):
    """
    Downloads a CSV file from Google Drive using its file ID to a temporary location,
    reads it into a pandas DataFrame, and returns the DataFrame.

    Args:
        file_id (str): The Google Drive file ID of the CSV.

    Returns:
        pd.DataFrame: The DataFrame loaded from the CSV file.
    """
    
        # Create a SparkSession if not already created.
    spark_sess = SparkSession.builder.getOrCreate()
    # Authenticate and download the file to a temporary location.
    local_temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    if download_file_from_drive(file_id, local_temp_file):
        try:
            df =  spark.read.csv(local_temp_file, header=True, inferSchema=True)
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
    else:
        print("Download failed.")
        return None