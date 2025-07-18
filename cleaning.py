import pandas as pd
import re
data=pd.read_csv('cscpopendata.csv')
# Handling missing values
columns_to_drop = ['ChemicalDateRemoved']
data = data.drop(columns=columns_to_drop)

# Fill missing values for specific columns
data['BrandName'] = data['BrandName'].fillna('Unknown Brand')
data['CasNumber'] = data['CasNumber'].fillna('Unknown CasNumber')
print(data['BrandName'].isnull().sum())

data['CSFId']=data['CSFId'].fillna(0)
data['CSFId']=data['CSFId'].astype('int')
data['CSF']=data['CSF'].fillna('Unknown')
data['CSF'] = data['CSF'].apply(lambda x: "Unknown" if str(x).isdigit() or re.match(r'^\d+(-\d+)+$', str(x)) else x)


# Parsing Date Fields
date_columns = ['InitialDateReported', 'MostRecentDateReported', 'ChemicalCreatedAt', 'ChemicalUpdatedAt']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')

# Formatting Categorical Variables
categorical_columns = ['PrimaryCategory', 'SubCategory', 'BrandName', 'CompanyName']
for col in categorical_columns:
    data[col] = data[col].str.strip().str.title()  # Remove extra spaces and standardize case

# Clean text in `ProductName`
data['ProductName'] = data['ProductName'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex=True)

# delete the duplicates
data.drop_duplicates(inplace=True)

data_cleaned=data
print("Rows before cleaning:", len(pd.read_csv('cscpopendata.csv')))
print("Rows after cleaning:", len(data_cleaned))
data_cleaned.to_csv('final_cleaned_data.csv', index=False)
print("Final cleaned data has been saved.")