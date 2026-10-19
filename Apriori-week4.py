import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("C:/Users/c3703803/Downloads/Online Retail.csv")

print(data.head())
print(data.info())
print(data.shape)
# Exploring the columns of the data
print(data.columns)
# Exploring the different regions of transactions
print(data.Country.unique())

print("--------Cleaning the Data----------")

# Stripping extra spaces in the description
data['Description'].str.strip()
print(data.Description)

# Dropping the rows without any invoice number
data.dropna(axis =0, subset =['InvoiceNo'], inplace = True)

# Convert invoice numbers to string
data['InvoiceNo']=data['InvoiceNo'].astype('str')

#dropping all transactions done on credit with the letter C
data = data[~data['InvoiceNo'].str.contains('C')]
print(data)