#!/usr/bin/env python
# coding: utf-8
"""
@author: HEMANT
"""

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


pd.set_option('display.max_columns', None)


# In[4]:


df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()


# In[22]:


df.head()


# ### property_type vs price

# In[10]:


sns.barplot(x=df['property_type'], y=df['price'], estimator=np.median)


# In[12]:


sns.boxplot(x=df['property_type'], y=df['price'])


# ### property_type vs area

# In[14]:


sns.barplot(x=df['property_type'], y=df['built_up_area'], estimator=np.median)


# In[15]:


sns.boxplot(x=df['property_type'], y=df['built_up_area'])


# In[19]:


# removing that crazy outlier
df = df[df['built_up_area'] != 737147]


# In[20]:


sns.boxplot(x=df['property_type'], y=df['built_up_area'])


# ### property_type vs price_per_sqft

# In[24]:


sns.barplot(x=df['property_type'], y=df['price_per_sqft'], estimator=np.median)


# In[25]:


sns.boxplot(x=df['property_type'], y=df['price_per_sqft'])


# In[32]:


# check outliers
df[df['price_per_sqft'] > 100000][['property_type','society','sector','price','price_per_sqft','area','areaWithType', 'super_built_up_area', 'built_up_area', 'carpet_area']]


# In[33]:


df.head()


# In[34]:


sns.heatmap(pd.crosstab(df['property_type'],df['bedRoom']))


# In[38]:


# checking outliers
df[df['bedRoom'] >= 10]


# In[39]:


sns.barplot(x=df['property_type'],y=df['floorNum'])


# In[40]:


sns.boxplot(x=df['property_type'],y=df['floorNum'])


# In[49]:


# checking for outliers
df[(df['property_type'] == 'house') & (df['floorNum'] > 10)]


# In[ ]:


# conclusion houses(villa) but in appartments


# In[50]:


df.head()


# In[51]:


sns.heatmap(pd.crosstab(df['property_type'],df['agePossession']))


# In[54]:


sns.heatmap(pd.pivot_table(df,index='property_type',columns='agePossession',values='price',aggfunc='mean'),annot=True)


# In[115]:


plt.figure(figsize=(15,4))
sns.heatmap(pd.pivot_table(df,index='property_type',columns='bedRoom',values='price',aggfunc='mean'),annot=True)


# In[57]:


sns.heatmap(pd.crosstab(df['property_type'],df['furnishing_type']))


# In[58]:


sns.heatmap(pd.pivot_table(df,index='property_type',columns='furnishing_type',values='price',aggfunc='mean'),annot=True)


# In[59]:


sns.barplot(x=df['property_type'],y=df['luxury_score'])


# In[60]:


sns.boxplot(x=df['property_type'],y=df['luxury_score'])


# In[61]:


df.head()


# In[65]:


# sector analysis
plt.figure(figsize=(15,6))
sns.heatmap(pd.crosstab(df['property_type'],df['sector'].sort_index()))


# In[79]:


# sector analysis
import re
# Group by 'sector' and calculate the average price
avg_price_per_sector = df.groupby('sector')['price'].mean().reset_index()

# Function to extract sector numbers
def extract_sector_number(sector_name):
    match = re.search(r'\d+', sector_name)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return a large number for non-numbered sectors

avg_price_per_sector['sector_number'] = avg_price_per_sector['sector'].apply(extract_sector_number)

# Sort by sector number
avg_price_per_sector_sorted_by_sector = avg_price_per_sector.sort_values(by='sector_number')

# Plot the heatmap
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sector_sorted_by_sector.set_index('sector')[['price']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Price per Sector (Sorted by Sector Number)')
plt.xlabel('Average Price')
plt.ylabel('Sector')
plt.show()


# In[80]:


avg_price_per_sqft_sector = df.groupby('sector')['price_per_sqft'].mean().reset_index()

avg_price_per_sqft_sector['sector_number'] = avg_price_per_sqft_sector['sector'].apply(extract_sector_number)

# Sort by sector number
avg_price_per_sqft_sector_sorted_by_sector = avg_price_per_sqft_sector.sort_values(by='sector_number')

# Plot the heatmap
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sqft_sector_sorted_by_sector.set_index('sector')[['price_per_sqft']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Sector (Sorted by Sector Number)')
plt.xlabel('Average Price per sqft')
plt.ylabel('Sector')
plt.show()


# In[82]:


luxury_score = df.groupby('sector')['luxury_score'].mean().reset_index()

luxury_score['sector_number'] = luxury_score['sector'].apply(extract_sector_number)

# Sort by sector number
luxury_score_sector = luxury_score.sort_values(by='sector_number')

# Plot the heatmap
plt.figure(figsize=(5, 25))
sns.heatmap(luxury_score_sector.set_index('sector')[['luxury_score']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Sector (Sorted by Sector Number)')
plt.xlabel('Average Price per sqft')
plt.ylabel('Sector')
plt.show()


# In[83]:


df.head()


# ### price

# In[88]:


plt.figure(figsize=(12,8))
sns.scatterplot(df[df['area']<10000]['area'],df['price'],hue=df['bedRoom'])


# In[89]:


plt.figure(figsize=(12,8))
sns.scatterplot(df[df['area']<10000]['area'],df['price'],hue=df['agePossession'])


# In[92]:


plt.figure(figsize=(12,8))
sns.scatterplot(df[df['area']<10000]['area'],df['price'],hue=df['furnishing_type'].astype('category'))


# In[95]:


sns.barplot(x=df['bedRoom'],y=df['price'],estimator=np.median)


# In[98]:


sns.barplot(x=df['agePossession'],y=df['price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[100]:


sns.barplot(x=df['agePossession'],y=df['area'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[101]:


sns.barplot(x=df['furnishing_type'],y=df['price'],estimator=np.median)


# In[102]:


sns.scatterplot(df['luxury_score'],df['price'])


# ### correlation

# In[107]:


plt.figure(figsize=(8,8))
sns.heatmap(df.corr())


# In[111]:


df.corr()['price'].sort_values(ascending=False)


# In[112]:


df.head()


# In[114]:


sns.pairplot(df)


# In[ ]:




