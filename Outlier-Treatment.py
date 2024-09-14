#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: HEMANT
"""

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.copy_on_write = True

# In[111]:


pd.set_option('display.max_columns', None)


# In[112]:


df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()


# In[113]:


df.head()


# In[114]:


df.shape


# In[115]:


df.columns


# In[116]:


# outliers on the basis of price column
sns.histplot(df['price'],kde=True)


# In[117]:


sns.boxplot(x=df['price'])


# In[118]:


# Calculate the IQR for the 'price' column
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

# Displaying the number of outliers and some statistics
num_outliers = outliers.shape[0]
outliers_price_stats = outliers['price'].describe()

num_outliers, outliers_price_stats


# In[119]:


outliers.sort_values('price',ascending=False).head(20)


# In[120]:


# on the basis of price col we can say that there are some genuine outliers but there are some data erros as well


# ### Price_per_sqft

# In[121]:


sns.histplot(df['price_per_sqft'],kde=True)


# In[122]:


sns.boxplot(x=df['price_per_sqft'])


# In[123]:


# Calculate the IQR for the 'price' column
Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

# Displaying the number of outliers and some statistics
num_outliers = outliers_sqft.shape[0]
outliers_sqft_stats = outliers_sqft['price_per_sqft'].describe()

num_outliers, outliers_sqft_stats


# In[124]:


outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)


# In[125]:


outliers_sqft['price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])


# In[126]:


outliers_sqft['price_per_sqft'].describe()


# In[127]:


df.update(outliers_sqft)


# In[128]:


sns.histplot(df['price_per_sqft'],kde=True)


# In[129]:


sns.boxplot(x=df['price_per_sqft'])


# In[130]:


df[df['price_per_sqft']>50000]


# In[131]:


df = df[df['price_per_sqft'] <= 50000]


# In[132]:


sns.boxplot(x=df['price_per_sqft'])


# ### Area

# In[167]:


sns.histplot(df['area'],kde=True)


# In[134]:


sns.boxplot(x=df['area'])


# In[135]:


df['area'].describe()


# In[136]:


df[df['area'] > 100000]


# In[142]:


df = df[df['area'] < 100000]


# In[143]:


sns.histplot(df['area'],kde=True)


# In[144]:


sns.boxplot(x=df['area'])


# In[149]:


df[df['area'] > 10000].sort_values('area',ascending=False)

# 818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471


# In[150]:


df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)


# In[151]:


df[df['area'] > 10000].sort_values('area',ascending=False)


# In[162]:


df.loc[48,'area'] = 115*9
df.loc[300,'area'] = 7250
df.loc[2666,'area'] = 5800
df.loc[1358,'area'] = 2660
df.loc[3195,'area'] = 2850
df.loc[2131,'area'] = 1812
df.loc[3088,'area'] = 2160
df.loc[3444,'area'] = 1175


# In[163]:


sns.histplot(df['area'],kde=True)


# In[164]:


sns.boxplot(x=df['area'])


# In[168]:


df['area'].describe()


# ### Bedroom

# In[171]:


sns.histplot(df['bedRoom'],kde=True)


# In[172]:


sns.boxplot(x=df['bedRoom'])


# In[173]:


df['bedRoom'].describe()


# In[192]:


df[df['bedRoom'] > 10].sort_values('bedRoom',ascending=False)


# In[193]:


df = df[df['bedRoom'] <= 10]


# In[194]:


df.shape


# In[195]:


sns.histplot(df['bedRoom'],kde=True)


# In[196]:


sns.boxplot(x=df['bedRoom'])


# In[197]:


df['bedRoom'].describe()


# ### Bathroom

# In[199]:


sns.histplot(df['bathroom'],kde=True)


# In[200]:


sns.boxplot(x=df['bathroom'])


# In[201]:


df[df['bathroom'] > 10].sort_values('bathroom',ascending=False)


# In[202]:


df.head()


# ### super built up area

# In[204]:


sns.histplot(df['super_built_up_area'],kde=True)


# In[205]:


sns.boxplot(x=df['super_built_up_area'])


# In[206]:


df['super_built_up_area'].describe()


# In[207]:


df[df['super_built_up_area'] > 6000]


# ### built up area

# In[209]:


sns.histplot(df['built_up_area'],kde=True)


# In[211]:


sns.boxplot(x=df['built_up_area'])


# In[212]:


df[df['built_up_area'] > 10000]


# ### carpet area

# In[226]:


sns.histplot(df['carpet_area'],kde=True)


# In[225]:


sns.boxplot(x=df['carpet_area'])


# In[216]:


df[df['carpet_area'] > 10000]


# In[223]:


df.loc[2131,'carpet_area'] = 1812


# In[224]:


df[df['carpet_area'] > 10000]


# In[227]:


df.head()


# In[228]:


sns.histplot(df['luxury_score'],kde=True)


# In[229]:


sns.boxplot(df['luxury_score'])


# In[230]:


df.shape


# In[234]:


df['price_per_sqft'] = round((df['price']*10000000)/df['area'])


# In[235]:


df.head()


# In[236]:


sns.histplot(df['price_per_sqft'],kde=True)


# In[237]:


sns.boxplot(df['price_per_sqft'])


# In[240]:


df[df['price_per_sqft'] > 42000]


# In[256]:


x = df[df['price_per_sqft'] <= 20000]
(x['area']/x['bedRoom']).quantile(0.02)


# In[259]:


df[(df['area']/df['bedRoom'])<183]


# In[ ]:




