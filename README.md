# datasci_9_data_prep
This is a repository for Assignment 8 in HHA507, ML Prep. 

I attempted to perform ML preparation for two datasets. Dataset 1 was unsuccessful while Dataset 2 was a lot more successful in terms of functionality. 

## Dataset 1
This dataset looked at the occurence and indication of chronic diseases in Metropolitan areas in the U.S. The aim I chose for this dataset was to look at sex as the independent variable with the rest being the dependent to see what factors can predict the sex to be male or female. 

### Extract 
I initially tried to prep for ML with this dataset. The result isn't what I wanted, but I attempted it nonetheless. 

[Originial link](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi)

### Transform
This dataset was massive, so much so that it was causing issues in Google Cloud Shell. For the following tasks, whenever I tried to transform the data, 
it would take a long time for Google Shell to give back the result. 

For example, when I did df.columns, the result would come out a couple minutes later. However, I continued to try to work with this dataset.

This line makes all the columns lowercase, and replaces white spaces with "_"
```
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns
```

I then dropped unncessary columns.
```
to_drop = [
    'datasource',
    'response',
    'locationdesc',
    'datavalueunit',
    'datavaluealt',
    'datavaluefootnotesymbol',
    'datavaluefootnote',
    'lowconfidencelimit',
    'highconfidencelimit',
    'stratificationcategory2',
    'stratification2',
    'stratificationcategory3',
    'stratification3',
    'geolocation',
    'responseid',
    'locationid',
    'topicid',
    'questionid',
    'datavaluetypeid',
    'stratificationcategoryid1',
    'stratificationid1',
    'stratificationcategoryid2',
    'stratificationid2',
    'stratificationcategoryid3',
    'stratificationid3'
]

df.drop(to_drop, axis=1, inplace=True, errors='ignore')
df.columns
```

I then attemmpted to do further data cleaning by using a for loop to remove specific values in a column. 
```
df['datavaluetype'].unique()

datavaluetypestodrop = [
    'Average Annual Number',
    'Crude Prevalence',
    'Age-adjusted Rate',
    'Local control of the regulation of alcohol outlet density',
    'Crude Rate',
    'Age-adjusted Prevalence',
    'US Dollars',
    'Commercial host (dram shop) liability status for alcohol service',
    'Per capita alcohol consumption',
    'Average Annual Age-adjusted Rate',
    'Average Annual Crude Rate',
    'Adjusted by age, sex, race and ethnicity',
    'Prevalence', 'Yes/No',
    'Percent', 'Mean',
    'Median',
    'Age-adjusted Mean'
]

for x in datavaluetypestodrop:
    df['datavaluetype'].drop(x)

df['datavaluetype'].unique()
```
This did not work. I used a for loop because the drop function for pandas doesn't take more than two values. I think this for loop didn't work because there
is something wrong with the way it is formatted. I still believe you can use a for loop as a method to drop desired values. What would work better
is to use the filter function in pandas instead, which is what I did for Dataset 2. I ignored further data cleaning in the interest of seeing what would happen
if I did not properly clean the dataset. 

Next, we turned each column into a dataset while turning it into ordinal scale. We perform this by doing the following code:
```
## perform ordinal encoding on yearstart_occ
enc = OrdinalEncoder()
enc.fit(df[['yearstart']])
df['yearstart'] = enc.transform(df[['yearstart']])

## create dataframe with mapping
df_mapping_yearstart = pd.DataFrame(enc.categories_[0], columns=['yearstart'])
df_mapping_yearstart['yearstart_ordinal'] = df_mapping_yearstart.index
df_mapping_yearstart

## save mapping to csv
df_mapping_yearstart.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_yearstart.csv', index=False)
```
The code looks at the first appearance of a value and assigns a number to it starting with 0. In a sense, it is indexing the first appearance of each value. This is performed so that a machine can better read a dataset. This code also creates a csv file in the processed folder within my data folder. 

I repeated this function for all the dependent and independent variables. This was done manually, but I believe this could be done using a function. 

The final line of the code creates a csv file of all the columns with ordinal scale as test data. 

```
df.sample(1000).to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/chronic.csv')
```

I had some issues as my code would just periodically stop. 

![](https://github.com/jas-tang/datasci_9_data_prep/blob/main/images/1.JPG)

This is where I ran into a problem with Google Cloud Shell. Since my dataset was massive, Google Cloud Shell would periodically crash when trying to run this many functions. Upon crashing, I would have to start over again as Google Cloud Shell runs on instances. This led to multiple hours of brute forcing my way through the dataset until I gave up and decided to drastically decrease the dataset. I started over, and used df.sample(5000) as my starting dataset, and the results of each function spit out instantly. In the future, I think I will move over to using Visual Studio Code as Google Cloud Shell is not reliable when it comes to crashes. 

This is what it looked like after doing the transform stage. 

![](https://github.com/jas-tang/datasci_9_data_prep/blob/main/images/2.JPG)

### Compute
After importing the appropriate packages, we began to create a model using our processed data. 

This specifies our dataset that has become ordinally scaled. 
```
df = pd.read_csv('/home/jason_tang/datasci_9_data_prep/data/processed/chronic.csv')
len(df)
```

This creates a standardize scale for our model.
```
scaler = StandardScaler()
scaler.fit(X) # Fit the scaler to the features
pickle.dump(scaler, open('/home/jason_tang/datasci_9_data_prep/data/models/scaler_100k.sav', 'wb'))
```


This trains our model based on our data.
```
# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('/home/jason_tang/datasci_9_data_prep/data/models/X_train_100k.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('/home/jason_tang/datasci_9_data_prep/data/models/X_columns_100k.sav', 'wb'))
```

We then created a baseline model and a Linear Regresion Model. 

```
##### Create a baseline model using DummyClassifier
# Initialize the DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
# Train the model on the training set
dummy.fit(X_train, y_train)
dummy_acc = dummy.score(X_val, y_val)



##### Create a Logistic Regression model
# Initialize the Logistic Regression model
log_reg = LogisticRegression()
# Train the model on the training set
log_reg.fit(X_train, y_train)
# Predict on the validation set
y_val_pred = log_reg.predict(X_val)
# Evaluate the model
log_reg_acc = log_reg.score(X_val, y_val)
log_reg_mse = mean_squared_error(y_val, y_val_pred)
log_reg_r2 = r2_score(y_val, y_val_pred)
# Print confusion matrix
print(confusion_matrix(y_val, y_val_pred))
# Display the classification report
print(classification_report(y_val, y_val_pred))
# Print the results
print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)
```

We then used xgboost to create another model. 
```
xgboost = XGBClassifier()
# Train the model on the training set
xgboost.fit(X_train, y_train)
# Predict on the validation set
y_val_pred = xgboost.predict(X_val)
# Evaluate the model
xgboost_acc = xgboost.score(X_val, y_val)
xgboost_mse = mean_squared_error(y_val, y_val_pred)
xgboost_r2 = r2_score(y_val, y_val_pred)
# Print confusion matrix
print(confusion_matrix(y_val, y_val_pred))
# Display the classification report
print(classification_report(y_val, y_val_pred))
# Print the results
print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)
print('XGBoost accuracy:', xgboost_acc)
print('XGBoost MSE:', xgboost_mse)
print('XGBoost R2:', xgboost_r2)
```

And then we started setting up hyperparameters. Hyperparameters are parameters whose values control the learning process and determine the values of model parameters that a learning algorithm ends up learning.
```

### now lets perform hyperparameter tuning on XGBoost
# Define the grid of hyperparameters
param_grid = {
    # there are 3 hyperparameters we want to tune:
    # and each hyperparameter has a list of values we want to try that need to be the same length
    # across all hyperparameters
    'learning_rate': [0.1, 0.01, 0.001], # learning rate is the step size shrinkage used to prevent overfitting
    'n_estimators': [100, 200, 300], # number of trees
    'max_depth': [3, 4, 5], # maximum depth of each tree
}

# Initialize the XGBoost classifier
xgboost = XGBClassifier()
# Initialize GridSearch
grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=3, n_jobs=-1)
# Fit the estimator
grid_search.fit(X_train, y_train)
# Predict on the validation set
y_val_pred = grid_search.predict(X_val)
# Evaluate the model
## Create dataframe of the actual and predicted values
df_results = pd.DataFrame({'actual': y_val, 'predicted': y_val_pred})
grid_search_acc = grid_search.score(X_val, y_val)
grid_search_mse = mean_squared_error(y_val, y_val_pred)
grid_search_r2 = r2_score(y_val, y_val_pred)
# Print confusion matrix
print(confusion_matrix(y_val, y_val_pred))
# Display the classification report
print(classification_report(y_val, y_val_pred))
# Print the results
print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('XGBoost Model 1 accuracy:', xgboost_acc)
print('XGBoost Model 2 accuracy:', grid_search_acc)
```

We print the best hyperparameters, and use those 
```
# Print the best parameters and the best score
print(grid_search.best_params_)
print(grid_search.best_score_)



### now lets use the best parameters to train a new model
# Initialize the XGBoost classifier
xgboost = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)
# Train the model on the training set
xgboost.fit(X_train, y_train)
# Predict on the test set
y_test_pred = xgboost.predict(X_test)
# Evaluate the model
xgboost_acc = xgboost.score(X_test, y_test)
```

Now we use the SHAP model and the LIME model to explain the predictions made by the ML model. 
```
#### feature importance with SHAP
explainer = shap.TreeExplainer(xgboost)
explanation = explainer(X_test)
shape_vlaues = explanation.values
shap.summary_plot(explanation, X_test, plot_type="bar")

##### feature importance with LIME
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns,
    class_names=['female', 'male'],
    mode='classification',
)

# Pick the observation in the validation set for which explanation is required
observation_1 = 20
# Get the explanation for Logistic Regression and show the prediction
exp = explainer.explain_instance(X_val[observation_1], xgboost.predict_proba, num_features=9)
exp.save_to_file('/home/jason_tang/datasci_9_data_prep/data/models/observation_1.html')
```

We also save the model itself.
```
pickle.dump(xgboost, open('/home/jason_tang/datasci_9_data_prep/data/models/xgboost_100k.sav', 'wb'))
```

This model probably does not work because the code was designed to predict a dependent variable with only two answers, 0 and 1. Since I did not properly perform the data cleaning portion correctly, there are more than two values for my dependent variable. 

### Testing
After importing proper packages, we begin to test our non functioning model with random data.

```
### try and load the model back
loaded_model = pickle.load(open('/home/jason_tang/datasci_9_data_prep/data/models/xgboost_100k.sav', 'rb'))
### load scaler
loaded_scaler = pickle.load(open('/home/jason_tang/datasci_9_data_prep/data/models/scaler_100k.sav', 'rb'))

## now lets create a new dataframe with the same column names and values
df_test = pd.DataFrame(columns=['yearstart', 'yearend', 'locationabbr', 'topic', 'question',
       'datavaluetype', 'datavalue', 'stratificationcategory1',
       'stratification1']
```

We choose a random ordinal number for each column to predict the dependent variable.
```
df_test.loc[0] = [0, 23, 32, 3, 0, 50, 0, 0, 0]
df_test_scaled = loaded_scaler.transform(df_test)
```

Now we predict using the test set. 
```
y_test_pred = loaded_model.predict(df_test_scaled)
# print value of prediction
print(y_test_pred[0])
```

Of course, I didn't get a result as my model is flawed. I remedied a lot of my issues with Dataset 1 on Dataset 2. Dataset 2 was a lot more successful. 

## Dataset 2

This dataset identified critical areas of research on the global Monkeypox outbreak in the interest of decreasing future outbreaks.
For my ML preparation, my intention was to predict whether the Agency and Office Name would be either the CDC or the DOE based off the rest of the information. The Agency was my indepedent variable. 

### Extract

[Original link](https://healthdata.gov/Health/Monkeypox-Research-Summary-Data/x7kq-cyv4)

This dataset was big, but not as large as Dataset 1. 

## Transform 

In fear of Google Cloud Shell crashing again, I decided to do my data tranforming in Google Colab in a Notebook. The result should still be the same.


I created a dataframe, and looked at the unqiue values for Agency and Office Names. 
```
df = pd.read_csv('/content/Monkeypox_Research_Summary_Data.csv')
df.sample(10)

df['Agency and Office Name'].unique()
```

I then filtered the dataframe that only included CDC and DOE.
```
selected_agencies = ['CDC', 'DOE']
filtered_df = df[df['Agency and Office Name'].isin(selected_agencies)]

filtered_df.columns
```

Then, similar to before, I dropped unwanted columns. 
```
todrop = [
    'Upcoming Milestones',
    'Anticipated Completion',
    'Brief Description',
    'Project Link']

filtered_df.drop(todrop, axis=1, inplace=True, errors='ignore')
```

I also cleaned and lowercased the columns.
```
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

Now, my dataset is much smaller. I then performed ordinal encoding and mapping, similar to Dataset 1. 
```
## perform ordinal encoding on research_activity
enc = OrdinalEncoder()
enc.fit(df[['research_activity']])
df['research_activity'] = enc.transform(df[['research_activity']])

## create dataframe with mapping
df_mapping_research_activity = pd.DataFrame(enc.categories_[0], columns=['research_activity'])
df_mapping_research_activity['research_activity_ordinal'] = df_mapping_research_activity.index
df_mapping_research_activity

## save mapping to csv
df_mapping_research_activity.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_research_activity.csv', index=False)
```
I repeated this process for the rest of the remaining columns. 

Finally, I saved a CSV of the dataset after the ordinal encoding and mapping.
```
df_mapping_status.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_status.csv', index=False)
```

### Compute
I performed the same preparation tasks for Dataset 1 as to Dataset 2. Please see my code for Dataset 2 to see what was done. 
In summary, we featured our independent variable, which now has two values.
```
# Define the features and the target variable
X = df.drop('agency_and_office_name', axis=1)  
y = df['agency_and_office_name']               
```

We then scaled the data and created a base and linear regression model. We used XGBoost the create another model, and set up and found viable hyperparameters. We then used SHAP and LIME to be able to interprete ML models. 

### Test

We load our created model back in. 
```
### try and load the model back
loaded_model = pickle.load(open('/home/jason_tang/datasci_9_data_prep/Data/models/xgboost_100k.sav.sav', 'rb'))
### load scaler
loaded_scaler = pickle.load(open('/home/jason_tang/datasci_9_data_prep/Data/models/scaler_100k.sav', 'rb'))
```

Then we create a dataframe that contains our dependent variables.
```
df_test = pd.DataFrame(columns=[['research_activity', 'project_title', 'topic','countries', 'status']]
```

We create a fake dataset and apply the scalar to it. 
```
## research_activity = 0 (Animal reservoirs)
## project_title = 0 (Assessing mpox infections in animals associated with human cases)
## topic = 0 (Epidemiology
## countries = 0 
## status = 0 


df_test.loc[0] = [0,0,0,0,0]
df_test_scaled = loaded_scaler.transform(df_test)
```

Then we run the prediction model. 
```
# Predict on the test set
y_test_pred = loaded_model.predict(df_test_scaled)
# print value of prediction
print(y_test_pred[0])
```

The result we got was [0], which corresponds with the CDC. The interpretation of this is that given the fake dataset of every ordinal scale being [0], meaning that it is an animal reservoir, its assessing human infection, topic of epidemiology, with no country or status given, there is a likelihood of the study being performed by the CDC. 

The dataset trained on was reduced to a very small amount, so its is highly possible that the model is not accurate. However, the model functions properly. 

![](https://github.com/jas-tang/datasci_9_data_prep/blob/main/images/3.JPG)
