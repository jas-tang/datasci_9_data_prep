import pandas as pandas

## Original link: https://healthdata.gov/Health/Monkeypox-Research-Summary-Data/x7kq-cyv4

df = pd.read_csv('file:///home/jason_tang/datasci_9_data_prep/Data/raw/Monkeypox_Research_Summary_Data.csv')
df.size
df.sample(5)