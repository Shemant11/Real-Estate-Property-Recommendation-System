# -*- coding: utf-8 -*-
"""
@author: HEMANT
"""

import numpy as np
import pandas as pd
import re

# Set pandas options to display all rows and columns in DataFrame output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the dataset from a CSV file
df = pd.read_csv('houses.csv')

# Display a random sample of 5 rows from the DataFrame
df.sample(5)

# Display the shape of the DataFrame (number of rows and columns)
df.shape

# Display information about the DataFrame (e.g., column types, non-null counts)
df.info()

# Check for duplicate rows and count them
df.duplicated().sum()

# Remove duplicate rows from the DataFrame
df = df.drop_duplicates()

# Display the shape of the DataFrame after removing duplicates
df.shape

# Check for missing values in each column
df.isnull().sum()

# Drop unnecessary columns from the DataFrame
df.drop(columns=['link', 'property_id'], inplace=True)

# Display the first few rows of the DataFrame after dropping columns
df.head()

# Rename column 'rate' to 'price_per_sqft' for clarity
df.rename(columns={'rate': 'price_per_sqft'}, inplace=True)

# Display the first few rows of the DataFrame after renaming the column
df.head()

# Display the counts of unique values in the 'society' column
df['society'].value_counts()

# Check the shape of the unique values in the 'society' column
df['society'].value_counts().shape

# Clean the 'society' column by removing numbers and special characters, then convert to lowercase
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

# Display the counts of unique values in the cleaned 'society' column
df['society'].value_counts().shape

# Replace 'nan' values in 'society' column with 'independent'
df['society'] = df['society'].str.replace('nan', 'independent')

# Display the first few rows of the DataFrame after replacing 'nan' values
df.head()

# Display the counts of unique values in the 'price' column
df['price'].value_counts()

# Remove rows where the price is 'Price on Request'
df = df[df['price'] != 'Price on Request']

# Display the first few rows of the DataFrame after removing 'Price on Request' rows
df.head()

# Function to convert price to numeric values
def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0]) / 100, 2)  # Convert 'Lac' to numeric value
        else:
            return round(float(x[0]), 2)  # Convert other prices to numeric value

# Apply the price conversion function to the 'price' column
df['price'] = df['price'].str.split(' ').apply(treat_price)

# Display the first 5 rows of the DataFrame after price conversion
df.head()

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

# Display the shape of the DataFrame after removing rows with missing 'bedRoom'
df.shape

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

# Fill missing values in 'additionalRoom' with 'not available' and convert to lowercase
df['additionalRoom'].fillna('not available', inplace=True)
df['additionalRoom'] = df['additionalRoom'].str.lower()

# Display the first few rows of the DataFrame after handling 'additionalRoom'
df.head()

# Display the counts of unique values in the 'noOfFloor' column
df['noOfFloor'].value_counts()

# Check for missing values in the 'noOfFloor' column
df['noOfFloor'].isnull().sum()

# Clean and convert 'noOfFloor' column to numeric values
df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)

# Display the first few rows of the DataFrame after converting 'noOfFloor'
df.head()

# Rename 'noOfFloor' column to 'floorNum' for consistency
df.rename(columns={'noOfFloor': 'floorNum'}, inplace=True)

# Display the first few rows of the DataFrame after renaming the column
df.head()

# Fill missing values in the 'facing' column with 'NA'
df['facing'].fillna('NA', inplace=True)

# Calculate and insert a new column 'area' based on 'price' and 'price_per_sqft'
df['area'] = round((df['price'] * 10000000) / df['price_per_sqft'])

# Insert a new column 'property_type' with a constant value 'house'
df.insert(loc=1, column='property_type', value='house')

# Display the first few rows and summary information of the cleaned DataFrame
df.head()
df.shape
df.info()

# Save the cleaned DataFrame to a new CSV file
df.to_csv('house_cleaned.csv', index=False)
