import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('/content/Monkeypox_Research_Summary_Data.csv')
df.sample(10)

df['Agency and Office Name'].unique()

df.columns

selected_agencies = ['CDC', 'DOE']
filtered_df = df[df['Agency and Office Name'].isin(selected_agencies)]

filtered_df.columns

todrop = [
    'Upcoming Milestones',
    'Anticipated Completion',
    'Brief Description',
    'Project Link']

filtered_df.drop(todrop, axis=1, inplace=True, errors='ignore')

filtered_df

filtered_df.to_csv('monkeypox.csv', index=False)

df = pd.read_csv('/home/jason_tang/datasci_9_data_prep/Data/raw/monkeypox.csv')
df.sample(5)

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

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


## perform ordinal encoding on project_title
enc = OrdinalEncoder()
enc.fit(df[['project_title']])
df['project_title'] = enc.transform(df[['project_title']])

## create dataframe with mapping
df_mapping_project_title = pd.DataFrame(enc.categories_[0], columns=['project_title'])
df_mapping_project_title['project_title_ordinal'] = df_mapping_project_title.index
df_mapping_project_title

## save mapping to csv
df_mapping_project_title.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_project_title.csv', index=False)


## perform ordinal encoding on topic
enc = OrdinalEncoder()
enc.fit(df[['topic']])
df['topic'] = enc.transform(df[['topic']])

## create dataframe with mapping
df_mapping_topic = pd.DataFrame(enc.categories_[0], columns=['topic'])
df_mapping_topic['topic_ordinal'] = df_mapping_topic.index
df_mapping_topic

## save mapping to csv
df_mapping_topic.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_topic.csv', index=False)



## perform ordinal encoding on agency_and_office_name
enc = OrdinalEncoder()
enc.fit(df[['agency_and_office_name']])
df['agency_and_office_name'] = enc.transform(df[['agency_and_office_name']])

## create dataframe with mapping
df_mapping_agency_and_office_name = pd.DataFrame(enc.categories_[0], columns=['agency_and_office_name'])
df_mapping_agency_and_office_name['agency_and_office_name_ordinal'] = df_mapping_agency_and_office_name.index
df_mapping_agency_and_office_name

## save mapping to csv
df_mapping_agency_and_office_name.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_agency_and_office_name.csv', index=False)



## perform ordinal encoding on countries
enc = OrdinalEncoder()
enc.fit(df[['countries']])
df['countries'] = enc.transform(df[['countries']])

## create dataframe with mapping
df_mapping_countries = pd.DataFrame(enc.categories_[0], columns=['countries'])
df_mapping_countries['countries'] = df_mapping_countries.index
df_mapping_countries

## save mapping to csv
df_mapping_countries.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_countries.csv', index=False)



## perform ordinal encoding on status
enc = OrdinalEncoder()
enc.fit(df[['status']])
df['status'] = enc.transform(df[['status']])

## create dataframe with mapping
df_mapping_status = pd.DataFrame(enc.categories_[0], columns=['status'])
df_mapping_status['status'] = df_mapping_status.index
df_mapping_status

## save mapping to csv
df_mapping_status.to_csv('/home/jason_tang/datasci_9_data_prep/Data/processed/mapping_status.csv', index=False)


