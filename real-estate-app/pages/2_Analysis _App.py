"""
@author: HEMANT
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Plotting Demo")

st.title('Analytics')

# Load the CSV file and feature text
new_df = pd.read_csv(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\datasets\data_viz1.csv")
feature_text = pickle.load(open(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\datasets\feature_text.pkl", 'rb'))

# Ensure that only numeric columns are considered for mean aggregation
numeric_cols = ['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']

# Convert the relevant columns to numeric to avoid issues
for col in numeric_cols:
    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

# Group by sector and calculate the mean for the numeric columns
group_df = new_df.groupby('sector')[numeric_cols].mean()

# Geomap of Price per Sqft
st.header('Sector Price per Sqft Geomap')
fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
                        color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                        mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)

st.plotly_chart(fig, use_container_width=True)

# Wordcloud for Features
st.header('Features Wordcloud')
wordcloud = WordCloud(width=800, height=800, background_color='black', stopwords=set(['s']),
                      min_font_size=10).generate(feature_text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
st.pyplot()

# Scatter Plot: Area Vs Price
st.header('Area Vs Price')
property_type = st.selectbox('Select Property Type', ['flat', 'house'])

if property_type == 'house':
    fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom",
                      title="Area Vs Price")
else:
    fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
                      title="Area Vs Price")

st.plotly_chart(fig1, use_container_width=True)

# Pie Chart: BHK Distribution
st.header('BHK Pie Chart')
sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0, 'overall')

selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':
    fig2 = px.pie(new_df, names='bedRoom')
else:
    fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

st.plotly_chart(fig2, use_container_width=True)

# Side by Side BHK Price Comparison
st.header('Side by Side BHK price comparison')
fig3 = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')
st.plotly_chart(fig3, use_container_width=True)

# Side by Side Distplot for Property Type
st.header('Side by Side Distplot for property type')
fig3 = plt.figure(figsize=(10, 4))
sns.distplot(new_df[new_df['property_type'] == 'house']['price'], label='house', hist=False)
sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat', hist=False)
plt.legend()
st.pyplot(fig3)
