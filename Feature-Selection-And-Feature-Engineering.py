#!/usr/bin/env python
# coding: utf-8
"""
@author: HEMANT
"""

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


pd.set_option('display.max_columns', None)


# In[16]:


df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')


# In[17]:


df.shape


# In[18]:


df.head()


# In[19]:


train_df = df.drop(columns=['society','price_per_sqft'])


# In[20]:


train_df.head()


# In[21]:


sns.heatmap(train_df.corr())


# In[22]:


train_df.corr()['price'].sort_values(ascending=False)


# In[13]:


# cols in question

# numerical -> luxury_score, others, floorNum
# categorical -> property_type, sector, agePossession


# In[ ]:




