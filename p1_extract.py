import pandas as pd

# Originial link: https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi
datalink = 'https://data.cdc.gov/api/views/g4ie-h725/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.size
df.sample(5)