# -*- coding: utf-8 -*-
"""
@author: HEMANT
"""

import numpy as np
import pandas as pd

# Load the cleaned data from CSV files into DataFrames
flats = pd.read_csv('flats_cleaned.csv')
houses = pd.read_csv('house_cleaned.csv')

# Concatenate the 'flats' and 'houses' DataFrames into a single DataFrame 'df'
# The 'ignore_index=True' parameter ensures that the index is reset in the resulting DataFrame
df = pd.concat([flats, houses], ignore_index=True)

# Shuffle the DataFrame rows randomly to mix the data
# The 'ignore_index=True' parameter ensures that the index is reset after shuffling
df = df.sample(df.shape[0], ignore_index=True)

# Display the first few rows of the combined and shuffled DataFrame
df.head()

# Save the combined DataFrame to a new CSV file named 'gurgaon_properties.csv'
# 'index=False' ensures that the index is not written to the file
df.to_csv('gurgaon_properties.csv', index=False)
