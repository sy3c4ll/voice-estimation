import os
import requests
from tqdm import tqdm
from zipfile import ZipFile

url = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zw4p4p7sdh-1.zip"
zipfile_path = "data/raw/" + url.split("/")[-1]
extract_path = "data/external/"

# Download the dataset
# Check extracted dataset existence
if not os.path.exists(extract_path + "".join(url.split("/")[-1].split(".")[:-1])):
  # Check raw archive existence
  if not os.path.exists(zipfile_path):
    print("[*] Downloading raw archive across HTTPS")
    response = requests.get(url, stream = True)
    with open(zipfile_path, "wb") as handle:
      for data in tqdm(response.iter_content(), total = int(response.headers['Content-Length'])):
        handle.write(data)
  else:
    print("[-] Raw dataset archive exists, skipping download")

  print("[*] Extracting archive")
  # Extract the dataset
  with ZipFile(zipfile_path, "r") as handle:
    handle.extractall(extract_path)
else:
  print("[-] Dataset directory exists, skipping")