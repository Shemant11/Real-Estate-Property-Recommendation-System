#!/usr/bin/env python
# coding: utf-8
"""
@author: HEMANT
"""

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


df = pd.read_csv('gurgaon_properties_cleaned_v2.csv')


# In[29]:


df.head()


# In[30]:


df.shape


# In[31]:


df.info()


# In[32]:


df.duplicated().sum()


# In[34]:


df.drop_duplicates(inplace=True)


# In[35]:


df.head()


# ### property_type

# In[40]:


df['property_type'].value_counts().plot(kind='bar',)


# #### Observations
# 
# - Flats are in majority(75 percent) and there are less number of houses(~25 percent)
# - No missing values

# ### society

# In[42]:


df['society'].value_counts().shape


# In[60]:


df['society'].value_counts()


# In[62]:


df[df['society'] != 'independent']['society'].value_counts(normalize=True).cumsum().head(75)


# In[63]:


society_counts = df['society'].value_counts()

# Frequency distribution for societies
frequency_bins = {
    "Very High (>100)": (society_counts > 100).sum(),
    "High (50-100)": ((society_counts >= 50) & (society_counts <= 100)).sum(),
    "Average (10-49)": ((society_counts >= 10) & (society_counts < 50)).sum(),
    "Low (2-9)": ((society_counts > 1) & (society_counts < 10)).sum(),
    "Very Low (1)": (society_counts == 1).sum()
}
frequency_bins


# In[47]:


# top 10 socities
df[df['society'] != 'independent']['society'].value_counts().head(10).plot(kind='bar')


# In[48]:


df['society'].isnull().sum()


# In[58]:


df[df['society'].isnull()]


# #### Observations
# 
# - Around 13% properties comes under independent tag.
# - There are 675 societies. 
# - The top 75 societies have 50 percent of the preperties and the rest 50 percent of the properties come under the remaining 600 societies
#     - Very High (>100): Only 1 society has more than 100 listings.
#     - High (50-100): 2 societies have between 50 to 100 listings.
#     - Average (10-49): 92 societies fall in this range with 10 to 49 listings each.
#     - Low (2-9): 273 societies have between 2 to 9 listings.
#     - Very Low (1): A significant number, 308 societies, have only 1 listing.
# - 1 missing value

# ### sector

# In[65]:


# unique sectors
df['sector'].value_counts().shape


# In[66]:


# top 10 sectors
df['sector'].value_counts().head(10).plot(kind='bar')


# In[67]:


# Frequency distribution for sectors
sector_counts = df['sector'].value_counts()

sector_frequency_bins = {
    "Very High (>100)": (sector_counts > 100).sum(),
    "High (50-100)": ((sector_counts >= 50) & (sector_counts <= 100)).sum(),
    "Average (10-49)": ((sector_counts >= 10) & (sector_counts < 50)).sum(),
    "Low (2-9)": ((sector_counts > 1) & (sector_counts < 10)).sum(),
    "Very Low (1)": (sector_counts == 1).sum()
}

sector_frequency_bins


# #### Observations
# 
# - There are a total of 104 unique sectors in the dataset.
# - Frequency distribution of sectors:
#     - Very High (>100): 3 sectors have more than 100 listings.
#     - High (50-100): 25 sectors have between 50 to 100 listings.
#     - Average (10-49): A majority, 60 sectors, fall in this range with 10 to 49 listings each.
#     - Low (2-9): 16 sectors have between 2 to 9 listings.
#     - Very Low (1): Interestingly, there are no sectors with only 1 listing.

# ### Price

# In[68]:


df['price'].isnull().sum()


# In[69]:


df['price'].describe()


# In[86]:


sns.histplot(df['price'], kde=True, bins=50)


# In[79]:


sns.boxplot(x=df['price'], color='lightgreen')
plt.grid()


# - Descriptive Statistics:
# 
#     - Count: There are 3,660 non-missing price entries.
#     - Mean Price: The average price is approximately 2.53 crores.
#     - Median Price: The median (or 50th percentile) price is 1.52 crores.
#     - Standard Deviation: The prices have a standard deviation of 2.98, indicating variability in the prices.
#     - Range: Prices range from a minimum of 0.07 crores to a maximum of 31.5 crores.
#     - IQR: The interquartile range (difference between 75th and 25th percentile) is from 0.95 crores to 2.75 crores.
# 
# 
# - Visualizations:
# 
#     - Distribution: The histogram indicates that most properties are priced in the lower range (below 5 crores), with a few properties going beyond 10 crores.
#     - Box Plot: The box plot showcases the spread of the data and potential outliers. Properties priced above approximately 10 crores might be considered outliers as they lie beyond the upper whisker of the box plot.
#  
# 
# - Missing Values: There are 17 missing values in the price column.

# In[88]:


# Skewness and Kurtosis
skewness = df['price'].skew()
kurtosis = df['price'].kurt()

print(skewness,kurtosis)


# **Skewness**: The price distribution has a skewness of approximately 3.28, indicating a positive skew. This means that the distribution tail is skewed to the right, which aligns with our observation from the histogram where most properties have prices on the lower end with a few high-priced properties.
# 
# **Kurtosis**: The kurtosis value is approximately 14.93. A kurtosis value greater than 3 indicates a distribution with heavier tails and more outliers compared to a normal distribution.

# In[89]:


# Quantile Analysis
quantiles = df['price'].quantile([0.01, 0.05, 0.95, 0.99])

quantiles


# Quantile Analysis:
# 
# - 1% Quantile: Only 1% of properties are priced below 0.25 crores.
# - 5% Quantile: 5% of properties are priced below 0.37 crores.
# - 95% Quantile: 95% of properties are priced below 8.5 crores.
# - 99% Quantile: 99% of properties are priced below 15.26 crores, indicating that very few properties are priced above this value.

# In[90]:


# Identify potential outliers using IQR method
Q1 = df['price'].describe()['25%']
Q3 = df['price'].describe()['75%']
IQR = Q3 - Q1

IQR


# In[91]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(lower_bound, upper_bound)


# In[95]:


outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
outliers.shape


# In[96]:


outliers['price'].describe()


# Outliers Analysis (using IQR method):
# 
# - Based on the IQR method, there are 425 properties considered as outliers.
# - These outliers have an average price of approximately 9.24 crores.
# - The range for these outliers is from 5.46 crores to 31.5 crores.

# In[98]:


# price binning
bins = [0, 1, 2, 3, 5, 10, 20, 50]
bin_labels = ["0-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-50"]
pd.cut(df['price'], bins=bins, labels=bin_labels, right=False).value_counts().sort_index().plot(kind='bar')


# - The majority of properties are priced in the "1-2 crores" and "2-3 crores" ranges.
# - There's a significant drop in the number of properties priced above "5 crores."

# In[101]:


# ecdf plot
ecdf = df['price'].value_counts().sort_index().cumsum() / len(df['price'])
plt.plot(ecdf.index, ecdf, marker='.', linestyle='none')
plt.grid()


# In[175]:


plt.figure(figsize=(15, 6))

# Distribution plot without log transformation
plt.subplot(1, 2, 1)
sns.histplot(df['price'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Prices (Original)')
plt.xlabel('Price (in Crores)')
plt.ylabel('Frequency')

# Distribution plot with log transformation
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), kde=True, bins=50, color='lightgreen')
plt.title('Distribution of Prices (Log Transformed)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# np.log1p(x): This function computes the natural logarithm of 1+x. 
# It's designed to provide more accurate results for values of x that are very close to zero.
# 
# Using np.log1p helps in transforming the price column while ensuring that any value (including zero, if present) is handled appropriately. When we need to reverse the transformation, we can use np.expm1 which computes e^x-1

# In[104]:


skewness = np.log1p(df['price']).skew()
kurtosis = np.log1p(df['price']).kurt()

print(skewness,kurtosis)


# In[106]:


plt.figure(figsize=(15, 6))

# Distribution plot without log transformation
plt.subplot(1, 2, 1)
sns.boxplot(df['price'], color='skyblue')
plt.title('Distribution of Prices (Original)')
plt.xlabel('Price (in Crores)')
plt.ylabel('Frequency')

# Distribution plot with log transformation
plt.subplot(1, 2, 2)
sns.boxplot(np.log1p(df['price']), color='lightgreen')
plt.title('Distribution of Prices (Log Transformed)')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# ### price_per_sqft

# In[112]:


df['price_per_sqft'].isnull().sum()


# In[113]:


df['price_per_sqft'].describe()


# In[109]:


sns.histplot(df['price_per_sqft'], bins=50, color='skyblue', kde=True)


# Most properties have a price_per_sqft ranging between approximately ₹0 and ₹40,000. There is a significant concentration in the lower range, with a few properties having exceptionally high price_per_sqft.

# In[110]:


sns.boxplot(df['price_per_sqft'], color='lightgreen')


# The box plot clearly shows several outliers, especially on the higher side. The interquartile range (IQR) is relatively compact, but there are many data points beyond the "whiskers" of the box plot, indicating potential outliers

# #### Observations
# 
# - Potential Outliers
# - Right Skewed
# - 17 missing values

# ### bedRoom

# In[115]:


df['bedRoom'].isnull().sum()


# In[117]:


df['bedRoom'].value_counts().sort_index().plot(kind='bar')


# In[121]:


df['bedRoom'].value_counts(normalize=True).head().plot(kind='pie',autopct='%0.2f%%')


# ### bathroom

# In[123]:


df['bathroom'].isnull().sum()


# In[124]:


df['bathroom'].value_counts().sort_index().plot(kind='bar')


# In[125]:


df['bathroom'].value_counts(normalize=True).head().plot(kind='pie',autopct='%0.2f%%')


# In[126]:


df.head()


# ### balcony

# In[127]:


df['balcony'].isnull().sum()


# In[129]:


df['balcony'].value_counts().plot(kind='bar')


# In[130]:


df['balcony'].value_counts(normalize=True).head().plot(kind='pie',autopct='%0.2f%%')


# In[138]:


### floorNum


# In[178]:


df.iloc[:,10:].head()


# In[139]:


df['floorNum'].isnull().sum()


# In[140]:


df['floorNum'].describe()


# In[141]:


df['floorNum'].value_counts().sort_index().plot(kind='bar')


# In[142]:


sns.boxplot(df['floorNum'], color='lightgreen')


# - The majority of the properties lie between the ground floor (0) and the 25th floor.
# - Floors 1 to 4 are particularly common, with the 3rd floor being the most frequent.
# - There are a few properties located at higher floors, but their frequency is much lower.
# - The box plot reveals that the majority of the properties are concentrated around the lower floors. The interquartile range (IQR) lies between approximately the 2nd and 10th floors.
# - Data points beyond the "whiskers" of the box plot, especially on the higher side, indicate potential outliers.

# ### facing

# In[144]:


df['facing'].isnull().sum()


# In[145]:


df['facing'].fillna('NA',inplace=True)


# In[146]:


df['facing'].value_counts()


# ### agePossession

# In[148]:


df['agePossession'].isnull().sum()


# In[149]:


df['agePossession'].value_counts()


# ### areas

# In[150]:


# super built up area
df['super_built_up_area'].isnull().sum()


# In[151]:


df['super_built_up_area'].describe()


# In[153]:


sns.histplot(df['super_built_up_area'].dropna(), bins=50, color='skyblue', kde=True)


# In[154]:


sns.boxplot(df['super_built_up_area'].dropna(), color='lightgreen')


# - Most properties have a super built-up area ranging between approximately 1,000 sq.ft and 2,500 sq.ft.
# - There are a few properties with a significantly larger area, leading to a right-skewed distribution.
# - The interquartile range (IQR) lies between roughly 1,480 sq.ft and 2,215 sq.ft, indicating that the middle 50% of the properties fall within this range.
# - There are several data points beyond the upper "whisker" of the box plot, indicating potential outliers. These are properties with an unusually large super built-up area.

# In[155]:


# built up area
df['built_up_area'].isnull().sum()


# In[156]:


df['built_up_area'].describe()


# In[157]:


sns.histplot(df['built_up_area'].dropna(), bins=50, color='skyblue', kde=False)


# In[158]:


sns.boxplot(df['built_up_area'].dropna(), color='lightgreen')


# - Most properties have a built-up area ranging roughly between 500 sq.ft and 3,500 sq.ft.
# - There are very few properties with a much larger built-up area, leading to a highly right-skewed distribution.
# - The box plot confirms the presence of significant outliers on the higher side. The data's interquartile range (IQR) is relatively compact, but the "whiskers" of the box plot are stretched due to the outliers.
# 
# 
# The presence of extreme values, especially on the higher side, suggests that there may be outliers or data errors. This could also be due to some properties being exceptionally large, like a commercial complex or an entire building being listed.

# In[159]:


# carpet area
df['carpet_area'].isnull().sum()


# In[160]:


df['carpet_area'].describe()


# In[161]:


sns.histplot(df['carpet_area'].dropna(), bins=50, color='skyblue', kde=False)


# In[162]:


sns.boxplot(df['carpet_area'].dropna(), color='lightgreen')


# In[164]:


df.iloc[:,16:]


# ### additional rooms

# In[167]:


plt.figure(figsize=(20, 12))

# Create a subplot of pie charts for each room type
for idx, room in enumerate(['study room','servant room','store room','pooja room','others'], 1):
    ax = plt.subplot(2, 3, idx)
    df[room].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    plt.title(f'Distribution of {room.title()}')
    plt.ylabel('')

plt.tight_layout()
plt.show()


# ### furnishing_type

# In[168]:


df['furnishing_type'].value_counts()


# In[169]:


df['furnishing_type'].value_counts().plot(kind='pie',autopct='%0.2f%%')


# ### luxury score

# In[170]:


df['luxury_score'].isnull().sum()


# In[171]:


df['luxury_score'].describe()


# In[172]:


sns.histplot(df['luxury_score'], bins=50, color='skyblue', kde=True)


# In[173]:


sns.boxplot(df['luxury_score'], color='lightgreen')


# The luxury score distribution has multiple peaks, suggesting a multi-modal distribution. There's a significant number of properties with lower luxury scores (around 0-50), and another peak is observed around the 110-130 range.
# 
# The box plot reveals that the majority of the properties have luxury scores between approximately 30 and 110. The interquartile range (IQR) lies between these values.

# In[179]:


df.head()


# In[ ]:




