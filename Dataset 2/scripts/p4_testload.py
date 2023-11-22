import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

### try and load the model back
loaded_model = pickle.load(open('/home/jason_tang/datasci_9_data_prep/Data/models/xgboost_100k.sav.sav', 'rb'))
### load scaler
loaded_scaler = pickle.load(open('/home/jason_tang/datasci_9_data_prep/Data/models/scaler_100k.sav', 'rb'))

## now lets create a new dataframe with the same column names and values
df_test = pd.DataFrame(columns=[['research_activity', 'project_title', 'topic','countries', 'status']])

## research_activity = 0 (Animal reservoirs)
## project_title = 0 (Assessing mpox infections in animals associated with human cases)
## topic = 0 (Epidemiology
## countries = 0 
## status = 0 


df_test.loc[0] = [0,0,0,0,0]
df_test_scaled = loaded_scaler.transform(df_test)

# Predict on the test set
y_test_pred = loaded_model.predict(df_test_scaled)
# print value of prediction
print(y_test_pred[0])

#Result is 0, which corresponds with CDC