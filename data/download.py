import requests
import zipfile
import os

def download_and_unzip_data(url, output_zip, extract_to='.'):
    # Download the file
    response = requests.get(url, stream=True)
    with open(output_zip, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)
    print('Download complete.')

    # Unzip the file
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print('Unzipping complete.')
