"""
@author: HEMANT
"""
import numpy as np
import pandas as pd
import re

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv('flats.csv')

# Display a random sample of 5 rows from the DataFrame
df.sample(5)

# Display the shape of the DataFrame (number of rows and columns)
df.shape

# Display information about the DataFrame (e.g., column types, non-null counts)
df.info()

# Check for duplicate rows and count them
df.duplicated().sum()

# Check for missing values in each column
df.isnull().sum()

# Drop unnecessary columns from the DataFrame
df.drop(columns=['link', 'property_id'], inplace=True)

# Display the first few rows of the DataFrame
df.head()

# Rename columns for better clarity
df.rename(columns={'area': 'price_per_sqft'}, inplace=True)
df.head()

# Display the counts of unique values in the 'society' column
df['society'].value_counts()

# Check the shape of the unique values in the 'society' column
df['society'].value_counts().shape

# Clean the 'society' column by removing numbers and special characters, and convert to lowercase
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

# Display the counts of unique values in the cleaned 'society' column
df['society'].value_counts().shape
df['society'].value_counts()
df.head()

# Display the counts of unique values in the 'price' column
df['price'].value_counts()

# Filter out rows where the price is 'Price on Request'
df[df['price'] == 'Price on Request']
df = df[df['price'] != 'Price on Request']
df.head()

# Function to convert price to a numeric value
def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100, 2)  # Convert Lac to numeric value
        else:
            return round(float(x[0]), 2)  # Convert other prices to numeric value

# Apply the price conversion function to the 'price' column
df['price'] = df['price'].str.split(' ').apply(treat_price)

# Display the first 5 rows of the DataFrame after price conversion
df.head(5)

# Display the counts of unique values in the 'price_per_sqft' column
df['price_per_sqft'].value_counts()

# Clean and convert the 'price_per_sqft' column to numeric values
df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹', '').str.replace(',', '').str.strip().astype('float')

# Display the first few rows of the DataFrame after cleaning 'price_per_sqft'
df.head()

# Display the counts of unique values in the 'bedRoom' column
df['bedRoom'].value_counts()

# Check for missing values in the 'bedRoom' column
df[df['bedRoom'].isnull()]

# Remove rows with missing values in the 'bedRoom' column
df = df[~df['bedRoom'].isnull()]

# Convert 'bedRoom' column to integer values
df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

# Display the first few rows of the DataFrame after converting 'bedRoom'
df.head()

# Display the counts of unique values in the 'bathroom' column
df['bathroom'].value_counts()

# Check for missing values in the 'bathroom' column
df['bathroom'].isnull().sum()

# Convert 'bathroom' column to integer values
df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

# Display the first few rows of the DataFrame after converting 'bathroom'
df.head()

# Display the counts of unique values in the 'balcony' column
df['balcony'].value_counts()

# Check for missing values in the 'balcony' column
df['balcony'].isnull().sum()

# Convert 'balcony' column to numeric values, replacing 'No' with 0
df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

# Display the first few rows of the DataFrame after converting 'balcony'
df.head()

# Display the counts of unique values in the 'additionalRoom' column
df['additionalRoom'].value_counts()
df['additionalRoom'].value_counts().shape

# Check for missing values in the 'additionalRoom' column and fill with 'not available'
df['additionalRoom'].isnull().sum()
df['additionalRoom'].fillna('not available', inplace=True)
df['additionalRoom'] = df['additionalRoom'].str.lower()

# Display the first few rows of the DataFrame after handling 'additionalRoom'
df.head()

# Display the 'floorNum' column and check for missing values
df['floorNum']
df['floorNum'].isnull().sum()
df[df['floorNum'].isnull()]

# Clean and convert the 'floorNum' column to numeric values, handling special cases
df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground', '0').str.replace('Basement', '-1').str.replace('Lower', '0').str.extract(r'(\d+)')

# Display the first few rows of the DataFrame after converting 'floorNum'
df.head()

# Display the counts of unique values in the 'facing' column
df['facing'].value_counts()

# Check for missing values in the 'facing' column and fill with 'NA'
df['facing'].isnull().sum()
df['facing'].fillna('NA', inplace=True)

# Display the counts of unique values in the 'facing' column after handling missing values
df['facing'].value_counts()

# Insert a new column 'area' calculated from 'price' and 'price_per_sqft'
df.insert(loc=4, column='area', value=round((df['price'] * 10000000) / df['price_per_sqft']))

# Insert a new column 'property_type' with a constant value 'flat'
df.insert(loc=1, column='property_type', value='flat')

# Display the first few rows and summary information of the cleaned DataFrame
df.head()
df.info()
df.shape

# Save the cleaned DataFrame to a new CSV file
df.to_csv('flats_cleaned.csv', index=False)






