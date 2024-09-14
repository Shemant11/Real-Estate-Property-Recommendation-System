#!/usr/bin/env python
# coding: utf-8
"""
@author: HEMANT
"""
# In[3]:


import pandas as pd
from pandas_profiling import ProfileReport

# Load your dataset
df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()

# Create the ProfileReport object
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# Generate the report
profile.to_file("output_report.html")





