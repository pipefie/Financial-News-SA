# Example snippet for Google Drive using PyDrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import json

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")


def authenticate_drive():
    """Authenticate with Google Drive using environment variables."""
    gauth = GoogleAuth()

    # Create settings dynamically
    gauth_settings = {
        "client_config_backend": "settings",
        "client_config": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": ["http://localhost"]
        },
        "oauth_scope": ["https://www.googleapis.com/auth/drive"],
    }

    # Save settings to a temporary file
    temp_settings_path = "/tmp/pydrive_settings.json"
    with open(temp_settings_path, "w") as f:
        json.dump(gauth_settings, f)

    gauth.LoadClientConfigFile(temp_settings_path)

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