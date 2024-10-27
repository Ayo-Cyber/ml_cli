import pandas as pd
import ssl
import urllib

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

data = pd.read_csv("https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/refs/heads/master/dataset.csv")
print(data.head())
