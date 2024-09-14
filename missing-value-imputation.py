#!/usr/bin/env python
# coding: utf-8
"""
@author: HEMANT
"""

# In[393]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[394]:


pd.set_option('display.max_columns', None)


# In[395]:


df = pd.read_csv('gurgaon_properties_outlier_treated.csv')


# In[396]:


df.head()


# In[397]:


df.isnull().sum()


# ### Built up area

# In[398]:


sns.scatterplot(df['built_up_area'],df['super_built_up_area'])


# In[399]:


sns.scatterplot(df['built_up_area'],df['carpet_area'])


# In[400]:


((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))


# In[401]:


all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]


# In[402]:


all_present_df.shape


# In[403]:


super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()


# In[404]:


carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()


# In[405]:


print(super_to_built_up_ratio, carpet_to_built_up_ratio)


# In[406]:


# both present built up null
sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]


# In[407]:


sbc_df.head()


# In[408]:


sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2),inplace=True)


# In[409]:


df.update(sbc_df)


# In[410]:


df.isnull().sum()


# In[411]:


# sb present c is null built up null
sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]


# In[412]:


sb_df.head()


# In[413]:


sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105),inplace=True)


# In[414]:


df.update(sb_df)


# In[415]:


df.isnull().sum()


# In[416]:


# sb null c is present built up null
c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]


# In[417]:


c_df.head()


# In[418]:


c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9),inplace=True)


# In[419]:


df.update(c_df)


# In[420]:


df.isnull().sum()


# In[421]:


sns.scatterplot(df['built_up_area'],df['price'])


# In[422]:


anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]


# In[423]:


anamoly_df.sample(5)


# In[424]:


anamoly_df['built_up_area'] = anamoly_df['area']


# In[425]:


df.update(anamoly_df)


# In[426]:


sns.scatterplot(df['built_up_area'],df['price'])


# In[427]:


df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'],inplace=True)


# In[428]:


df.head()


# In[429]:


df.isnull().sum()


# ### floorNum

# In[430]:


df[df['floorNum'].isnull()]


# In[431]:


df[df['property_type'] == 'house']['floorNum'].median()


# In[432]:


df['floorNum'].fillna(2.0,inplace=True)


# In[433]:


df.isnull().sum()


# In[434]:


1011/df.shape[0]


# ### facing

# In[435]:


df['facing'].value_counts().plot(kind='pie',autopct='%0.2f%%')


# In[436]:


df.drop(columns=['facing'],inplace=True)


# In[437]:


df.sample(5)


# In[438]:


df.isnull().sum()


# In[439]:


df.drop(index=[2536],inplace=True)


# In[440]:


df.isnull().sum()


# ### agePossession

# In[441]:


df['agePossession'].value_counts()


# In[462]:


df[df['agePossession'] == 'Undefined']


# In[460]:


def mode_based_imputation(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


# In[ ]:





# In[444]:


df['agePossession'] = df.apply(mode_based_imputation,axis=1)


# In[450]:


df['agePossession'].value_counts()


# In[451]:


def mode_based_imputation2(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


# In[454]:


df['agePossession'] = df.apply(mode_based_imputation2,axis=1)


# In[455]:


df['agePossession'].value_counts()


# In[456]:


def mode_based_imputation3(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


# In[457]:


df['agePossession'] = df.apply(mode_based_imputation3,axis=1)


# In[458]:


df['agePossession'].value_counts()


# In[459]:


df.isnull().sum()


# In[463]:


df.to_csv('gurgaon_properties_missing_value_imputation.csv',index=False)


# In[464]:


df.shape


# In[ ]:




