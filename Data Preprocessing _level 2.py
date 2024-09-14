# -*- coding: utf-8 -*-
"""
@author: HEMANT
"""

import numpy as np
import pandas as pd

# Set display options for pandas to show all rows and columns in the output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the dataset from a CSV file into a DataFrame
df = pd.read_csv('gurgaon_properties.csv')

# Display a random sample of 5 rows from the DataFrame
df.sample(5)

# Display the shape of the DataFrame (number of rows and columns)
df.shape

# Check for duplicate rows in the DataFrame and display the count
df.duplicated().sum()

# Display summary information about the DataFrame
df.info()

# Check for missing values in each column and display the count
df.isnull().sum()

# Display the first few rows of the DataFrame
df.head()

# Extract 'sector' information from the 'property_name' column and insert it as a new column
df.insert(loc=3, column='sector', value=df['property_name'].str.split('in').str.get(1).str.replace('Gurgaon', '').str.strip())

# Display the updated DataFrame
df.head()

# Convert the 'sector' column to lowercase for uniformity
df['sector'] = df['sector'].str.lower()

# Display the updated DataFrame
df.head()

# Display the count of unique values in the 'sector' column
df['sector'].value_counts()

# Replace various sector names with standardized sector names
df['sector'] = df['sector'].str.replace('dharam colony', 'sector 12')
df['sector'] = df['sector'].str.replace('krishna colony', 'sector 7')
df['sector'] = df['sector'].str.replace('suncity', 'sector 54')
df['sector'] = df['sector'].str.replace('prem nagar', 'sector 13')
df['sector'] = df['sector'].str.replace('mg road', 'sector 28')
df['sector'] = df['sector'].str.replace('gandhi nagar', 'sector 28')
df['sector'] = df['sector'].str.replace('laxmi garden', 'sector 11')
df['sector'] = df['sector'].str.replace('shakti nagar', 'sector 11')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('baldev nagar', 'sector 7')
df['sector'] = df['sector'].str.replace('shivpuri', 'sector 7')
df['sector'] = df['sector'].str.replace('garhi harsaru', 'sector 17')
df['sector'] = df['sector'].str.replace('imt manesar', 'manesar')
df['sector'] = df['sector'].str.replace('adarsh nagar', 'sector 12')
df['sector'] = df['sector'].str.replace('shivaji nagar', 'sector 11')
df['sector'] = df['sector'].str.replace('bhim nagar', 'sector 6')
df['sector'] = df['sector'].str.replace('madanpuri', 'sector 7')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('saraswati vihar', 'sector 28')
df['sector'] = df['sector'].str.replace('arjun nagar', 'sector 8')
df['sector'] = df['sector'].str.replace('ravi nagar', 'sector 9')
df['sector'] = df['sector'].str.replace('vishnu garden', 'sector 105')
df['sector'] = df['sector'].str.replace('bhondsi', 'sector 11')
df['sector'] = df['sector'].str.replace('surya vihar', 'sector 21')
df['sector'] = df['sector'].str.replace('devilal colony', 'sector 9')
df['sector'] = df['sector'].str.replace('valley view estate', 'gwal pahari')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('mehrauli  road', 'sector 14')
df['sector'] = df['sector'].str.replace('jyoti park', 'sector 7')
df['sector'] = df['sector'].str.replace('ansal plaza', 'sector 23')
df['sector'] = df['sector'].str.replace('dayanand colony', 'sector 6')
df['sector'] = df['sector'].str.replace('sushant lok phase 2', 'sector 55')
df['sector'] = df['sector'].str.replace('chakkarpur', 'sector 28')
df['sector'] = df['sector'].str.replace('greenwood city', 'sector 45')
df['sector'] = df['sector'].str.replace('subhash nagar', 'sector 12')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('sohna road road', 'sohna road')
df['sector'] = df['sector'].str.replace('malibu town', 'sector 47')
df['sector'] = df['sector'].str.replace('surat nagar 1', 'sector 104')
df['sector'] = df['sector'].str.replace('new colony', 'sector 7')
df['sector'] = df['sector'].str.replace('mianwali colony', 'sector 12')
df['sector'] = df['sector'].str.replace('jacobpura', 'sector 12')
df['sector'] = df['sector'].str.replace('rajiv nagar', 'sector 13')
df['sector'] = df['sector'].str.replace('ashok vihar', 'sector 3')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('dlf phase 1', 'sector 26')
df['sector'] = df['sector'].str.replace('nirvana country', 'sector 50')
df['sector'] = df['sector'].str.replace('palam vihar', 'sector 2')
df['sector'] = df['sector'].str.replace('dlf phase 2', 'sector 25')
df['sector'] = df['sector'].str.replace('sushant lok phase 1', 'sector 43')
df['sector'] = df['sector'].str.replace('laxman vihar', 'sector 4')
df['sector'] = df['sector'].str.replace('dlf phase 4', 'sector 28')
df['sector'] = df['sector'].str.replace('dlf phase 3', 'sector 24')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('sushant lok phase 3', 'sector 57')
df['sector'] = df['sector'].str.replace('dlf phase 5', 'sector 43')
df['sector'] = df['sector'].str.replace('rajendra park', 'sector 105')
df['sector'] = df['sector'].str.replace('uppals southend', 'sector 49')
df['sector'] = df['sector'].str.replace('sohna', 'sohna road')
df['sector'] = df['sector'].str.replace('ashok vihar phase 3 extension', 'sector 5')
df['sector'] = df['sector'].str.replace('south city 1', 'sector 41')
df['sector'] = df['sector'].str.replace('ashok vihar phase 2', 'sector 5')

# Filter out sectors that appear less than 3 times
a = df['sector'].value_counts()[df['sector'].value_counts() >= 3]
df = df[df['sector'].isin(a.index)]

# Display the updated count of unique sector values
df['sector'].value_counts()

# Replace remaining sector names with standardized names
df['sector'] = df['sector'].str.replace('sector 95a', 'sector 95')
df['sector'] = df['sector'].str.replace('sector 23a', 'sector 23')
df['sector'] = df['sector'].str.replace('sector 12a', 'sector 12')
df['sector'] = df['sector'].str.replace('sector 3a', 'sector 3')
df['sector'] = df['sector'].str.replace('sector 110 a', 'sector 110')
df['sector'] = df['sector'].str.replace('patel nagar', 'sector 15')
df['sector'] = df['sector'].str.replace('a block sector 43', 'sector 43')
df['sector'] = df['sector'].str.replace('maruti kunj', 'sector 12')
df['sector'] = df['sector'].str.replace('b block sector 43', 'sector 43')

# Continue replacing sector names with standardized names
df['sector'] = df['sector'].str.replace('sector-33 sohna road', 'sector 33')
df['sector'] = df['sector'].str.replace('sector 1 manesar', 'manesar')
df['sector'] = df['sector'].str.replace('sector 4 phase 2', 'sector 4')
df['sector'] = df['sector'].str.replace('sector 1a manesar', 'manesar')
df['sector'] = df['sector'].str.replace('c block sector 43', 'sector 43')
df['sector'] = df['sector'].str.replace('sector 89 a', 'sector 89')
df['sector'] = df['sector'].str.replace('sector 2 extension', 'sector 2')
df['sector'] = df['sector'].str.replace('sector 36 sohna road', 'sector 36')

# Check for rows where the sector is labeled 'new' and correct them
df[df['sector'] == 'new']
df.loc[955, 'sector'] = 'sector 37'
df.loc[2800, 'sector'] = 'sector 92'
df.loc[2838, 'sector'] = 'sector 90'
df.loc[2857, 'sector'] = 'sector 76'

# Check for rows where the sector is labeled 'new sector 2' and correct them
df[df['sector'] == 'new sector 2']
df.loc[[311, 1072, 1486, 3040, 3875], 'sector'] = 'sector 110'

# Display the shape of the DataFrame after cleaning
df.shape

# Check for duplicate rows after cleaning and display the count
df.duplicated().sum()

# Display the first few rows of the cleaned DataFrame
df.head()

# Drop unnecessary columns from the DataFrame
df.drop(columns=['property_name', 'address', 'description', 'rating'], inplace=True)

# Display a random sample of 5 rows from the cleaned DataFrame
df.sample(5)

# Check for duplicate rows again after cleaning
df.duplicated().sum()

# Save the cleaned DataFrame to a new CSV file
df.to_csv('gurgaon_properties_cleaned_v1.csv', index=False)
