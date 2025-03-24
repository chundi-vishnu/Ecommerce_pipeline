import kaggle
import pandas as pd
# Download the dataset
def download():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('olistbr/brazilian-ecommerce', path='./data', unzip=True)
print("dataset is downloaded")