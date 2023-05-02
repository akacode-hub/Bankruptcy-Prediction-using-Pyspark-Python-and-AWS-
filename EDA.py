#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.io import arff
from pyspark.ml.feature import Imputer, StandardScaler


# In[2]:


# Load the five years of data
year1 = pd.read_csv("csv_result-1year.csv")
year2 = pd.read_csv("csv_result-2year.csv")
year3 = pd.read_csv("csv_result-3year.csv")
year4 = pd.read_csv("csv_result-4year.csv")
year5 = pd.read_csv("csv_result-5year.csv")


# In[3]:


df_raw = pd.concat([year1, year2, year3, year4, year5], ignore_index=True)

df_raw


# In[4]:


df_raw.columns


# In[5]:


import pandas as pd
import numpy as np

float_cols = ['Attr' + str(i) for i in range(1, 65)]

# Replace '0' and '?' with NaN
df_raw[float_cols] = df_raw[float_cols].replace(['0', '?'], np.nan)

# Cast columns to float
df_raw[float_cols] = df_raw[float_cols].apply(pd.to_numeric, errors='coerce')

df_raw


# In[6]:


missing_values = df_raw.isna().sum()
print(missing_values)


# In[7]:


import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(15, 5))

plt.bar(missing_values.index, missing_values.values)
plt.title('Missing Values')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.show()


# In[8]:


missing_values = df_raw['Attr37'].isna().sum()
print(missing_values)


# In[ ]:





# In[9]:


#Removing the id and Atttr37 column
# select columns to drop
cols_to_drop = ['id']

# drop selected columns
df_raw = df_raw.drop(columns=cols_to_drop)
df_raw


# In[10]:


from sklearn.impute import SimpleImputer
import pandas as pd

# Create a copy of the original dataframe
imputed_df = df_raw.copy()

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Impute the missing values
imputed_df[float_cols] = imputer.fit_transform(imputed_df[float_cols])

# Drop unwanted columns
imputed_df.drop('Attr37', axis=1, inplace=True)


# Rename the target column
imputed_df = imputed_df.rename(columns={'class_imputed': 'label'})



# In[11]:


imputed_df.columns


# In[12]:


imputed_df['class'].value_counts()


# In[13]:


import matplotlib.pyplot as plt

class_counts = imputed_df['class'].value_counts()
labels = class_counts.index.tolist()
sizes = class_counts.values.tolist()

class_counts = imputed_df['class'].value_counts()
class_counts.plot(kind='barh')
plt.title('Class Distribution')
plt.xlabel('Count')
plt.ylabel('Class')
plt.show()

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
ax.set_title('Class Distribution')
plt.show()


# In[ ]:





# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Net Profit to Total Assets (Attr1)
cash_flow = imputed_df['Attr1'].loc[imputed_df['class'] == 1].values
sns.histplot(cash_flow, kde=True, stat="density", color='#FB8861', ax=axes[0, 0])
sns.kdeplot(cash_flow, color='blue', linewidth=2, ax=axes[0, 0])
axes[0, 0].set_title('Net Profit to Total Assets \n(Bankrupt companies)', fontsize=14)
axes[0, 0].set_xlabel('Net Profit to Total Assets')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_ylim(0, 0.2)

# Total Liabilities / Total Assets (Attr2)
cash_flow = imputed_df['Attr2'].loc[imputed_df['class'] == 1].values
sns.histplot(cash_flow, kde=True, stat="density", color='#FB8861', ax=axes[0, 1])
sns.kdeplot(cash_flow, color='blue', linewidth=2, ax=axes[0, 1])
axes[0, 1].set_title('Total Liabilities / Total Assets \n(Bankrupt companies)', fontsize=14)
axes[0, 1].set_xlabel('Total Liabilities / Total Assets')
axes[0, 1].set_ylabel('Density')

# Working Capital / Total Assets (Attr3)
cash_flow = imputed_df['Attr3'].loc[imputed_df['class'] == 1].values
sns.histplot(cash_flow, kde=True, stat="density", color='#FB8861', ax=axes[1, 0])
sns.kdeplot(cash_flow, color='blue', linewidth=2, ax=axes[1, 0])
axes[1, 0].set_title('Working Capital / Total Assets \n(Bankrupt companies)', fontsize=14)
axes[1, 0].set_xlabel('Working Capital / Total Assets')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_ylim(0, 0.25)

# Equity / Total Assets (Attr10)
cash_flow = imputed_df['Attr10'].loc[imputed_df['class'] == 1].values
sns.histplot(cash_flow, kde=True, stat="density", color='#FB8861', ax=axes[1, 1])
sns.kdeplot(cash_flow, color='blue', linewidth=2, ax=axes[1, 1])
axes[1, 1].set_title('Equity / Total Assets \n(Bankrupt companies)', fontsize=14)
axes[1, 1].set_xlabel('Equity / Total Assets')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_ylim(0, 0.15)

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()


# In[ ]:




