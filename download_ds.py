import os
import requests
import zipfile
import gdown


if __name__ == '__main__':
    # download DATASET

    file_id = '1aK6O4DOEqorKucEjI4Q7SfRYQlRPKzYh'

    # Specify the directory where you want to save the downloaded and unzipped files
    download_directory = "dataset"

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)

    # Construct the export link for downloading
    export_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file
    response = requests.get(export_url)
    output_zip = os.path.join(download_directory, 'dataset.zip')

    gdown.download(export_url, output_zip, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(download_directory)

    # remove the zip file after extracting
    os.remove(output_zip)

