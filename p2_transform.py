import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

datalink = 'https://data.cdc.gov/api/views/g4ie-h725/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df = df.sample(5000)
df
## get column names
df.columns

## do some data cleaning of column names, 
## make them all lower case, remove white spaces and replace with _ 
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## get data types
df.dtypes # nice combination of numbers and strings/objects 

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

#Attempted to do data cleaning

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


## perform ordinal encoding on yearend_occ
enc = OrdinalEncoder()
enc.fit(df[['yearend']])
df['yearend'] = enc.transform(df[['yearend']])

## create dataframe with mapping
df_mapping_yearend = pd.DataFrame(enc.categories_[0], columns=['yearend'])
df_mapping_yearend['yearend_ordinal'] = df_mapping_yearend.index
df_mapping_yearend

## save mapping to csv
df_mapping_yearend.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_yearend.csv', index=False)
    

## perform ordinal encoding on locationabbr_occ
enc = OrdinalEncoder()
enc.fit(df[['locationabbr']])
df['locationabbr'] = enc.transform(df[['locationabbr']])

## create dataframe with mapping
df_mapping_locationabbr = pd.DataFrame(enc.categories_[0], columns=['locationabbr'])
df_mapping_locationabbr['locationabbr_ordinal'] = df_mapping_locationabbr.index
df_mapping_locationabbr

## save mapping to csv
df_mapping_locationabbr.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_locationabbr.csv', index=False)


## perform ordinal encoding on topic_occ
enc = OrdinalEncoder()
enc.fit(df[['topic']])
df['topic'] = enc.transform(df[['topic']])

## create dataframe with mapping
df_mapping_topic = pd.DataFrame(enc.categories_[0], columns=['topic'])
df_mapping_topic['topic_ordinal'] = df_mapping_topic.index
df_mapping_topic

## save mapping to csv
df_mapping_topic.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_topic.csv', index=False)


## perform ordinal encoding on question_occ
enc = OrdinalEncoder()
enc.fit(df[['question']])
df['question'] = enc.transform(df[['question']])

## create dataframe with mapping
df_mapping_question = pd.DataFrame(enc.categories_[0], columns=['question'])
df_mapping_question['question_ordinal'] = df_mapping_question.index
df_mapping_question

## save mapping to csv
df_mapping_question.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_question.csv', index=False)


## perform ordinal encoding on datavaluetype_occ
enc = OrdinalEncoder()
enc.fit(df[['datavaluetype']])
df['datavaluetype'] = enc.transform(df[['datavaluetype']])

## create dataframe with mapping
df_mapping_datavaluetype = pd.DataFrame(enc.categories_[0], columns=['datavaluetype'])
df_mapping_datavaluetype['datavaluetype_ordinal'] = df_mapping_datavaluetype.index
df_mapping_datavaluetype

## save mapping to csv
df_mapping_datavaluetype.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_datavaluetype.csv', index=False)


## perform ordinal encoding on datavalue_occ
enc = OrdinalEncoder()
enc.fit(df[['datavalue']])
df['datavalue'] = enc.transform(df[['datavalue']])

## create dataframe with mapping
df_mapping_datavalue = pd.DataFrame(enc.categories_[0], columns=['datavalue'])
df_mapping_datavalue['datavalue_ordinal'] = df_mapping_datavalue.index
df_mapping_datavalue

## save mapping to csv
df_mapping_datavalue.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_datavalue.csv', index=False)

## perform ordinal encoding on stratificationcategory1_occ
enc = OrdinalEncoder()
enc.fit(df[['stratificationcategory1']])
df['stratificationcategory1'] = enc.transform(df[['stratificationcategory1']])

## create dataframe with mapping
df_mapping_stratificationcategory1 = pd.DataFrame(enc.categories_[0], columns=['stratificationcategory1'])
df_mapping_stratificationcategory1['stratificationcategory1_ordinal'] = df_mapping_stratificationcategory1.index
df_mapping_stratificationcategory1

## save mapping to csv
df_mapping_stratificationcategory1.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_stratificationcategory1.csv', index=False)


## perform ordinal encoding on stratification1_occ
enc = OrdinalEncoder()
enc.fit(df[['stratification1']])
df['stratification1'] = enc.transform(df[['stratification1']])

## create dataframe with mapping
df_mapping_stratification1 = pd.DataFrame(enc.categories_[0], columns=['stratification1'])
df_mapping_stratification1['stratification1_ordinal'] = df_mapping_stratification1.index
df_mapping_stratification1

## save mapping to csv
df_mapping_stratification1.to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/mapping_stratification1.csv', index=False)

df.sample(1000).to_csv('/home/jason_tang/datasci_9_data_prep/data/processed/chronic.csv')