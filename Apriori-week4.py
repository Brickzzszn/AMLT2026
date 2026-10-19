import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("C:/Users/c3703803/Downloads/Online Retail.csv")

print(data.head())
print(data.info())
