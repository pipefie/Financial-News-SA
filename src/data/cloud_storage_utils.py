# Example snippet for Google Drive using PyDrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def authenticate_drive():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)

def list_drive_files(folder_id):
    drive = authenticate_drive()
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return [(file['title'], file['id']) for file in file_list]

def create_folder(folder_name, parent_folder_id=None):
    drive = authenticate_drive()
    folder_metadata = {'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_folder_id:
        folder_metadata['parents'] = [{'id': parent_folder_id}]
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    print(f"Folder '{folder_name}' created with ID: {folder['id']}")
    return folder['id']

def create_or_update_file(file_path, file_title, folder_id=None):
    drive = authenticate_drive()
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
    query = f"title='{file_title}' and trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    file_list = drive.ListFile({'q': query}).GetList()
    if file_list:
        return file_list[0]['id']
    else:
        return None