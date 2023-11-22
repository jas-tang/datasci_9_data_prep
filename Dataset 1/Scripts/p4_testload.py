import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

### try and load the model back
loaded_model = pickle.load(open('/home/jason_tang/datasci_9_data_prep/data/models/xgboost_100k.sav', 'rb'))
### load scaler
loaded_scaler = pickle.load(open('/home/jason_tang/datasci_9_data_prep/data/models/scaler_100k.sav', 'rb'))

## now lets create a new dataframe with the same column names and values
df_test = pd.DataFrame(columns=['yearstart', 'yearend', 'locationabbr', 'topic', 'question',
       'datavaluetype', 'datavalue', 'stratificationcategory1',
       'stratification1']

## yearstart = 11 
## yearend = 0
## locationabbr = 10
## topic = 0
## question = 3
## datavaluetype = 0
## datavalue = 6
## stratifcationcategory = 0
## stratification1 = 152


df_test.loc[0] = [0, 23, 32, 3, 0, 50, 0, 0, 0]
df_test_scaled = loaded_scaler.transform(df_test)

# Predict on the test set
y_test_pred = loaded_model.predict(df_test_scaled)
# print value of prediction
print(y_test_pred[0])