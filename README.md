# datasci_9_data_prep
This is a repository for Assignment 8 in HHA507, ML Prep. 

I attempted to perform ML preparation for two datasets. Dataset 1 was unsuccessful while Dataset 2 was a lot more successful in terms of functionality. 

## Dataset 1
This dataset looked at the occurence and indication of chronic diseases in Metropolitan areas in the U.S. The aim I chose for this dataset was to look at sex as the independent variable with the rest being the dependent to see what factors can predict the sex to be male or female. 

### Extract 
I initially tried to prep for ML with this dataset. The result isn't what I wanted, but I attempted it nonetheless. 

Originial link: https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi

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

This is where I ran into a problem with Google Cloud Shell. Since my dataset was massive, Google Cloud Shell would periodically crash when trying to run this many functions. Upon crashing, I would have to start over again as Google Cloud Shell runs on instances. This led to multiple hours of brute forcing my way through the dataset until I gave up and decided to drastically decrease the dataset. I started over, and used df.sample(5000) as my starting dataset, and the results of each function spit out instantly. In the future, I think I will move over to using Visual Studio Code as Google Cloud Shell is not reliable when it comes to crashes. 
