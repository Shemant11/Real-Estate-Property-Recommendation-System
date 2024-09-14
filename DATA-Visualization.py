#!/usr/bin/env python
# coding: utf-8
"""
@author: HEMANT
"""

# In[117]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[118]:


df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')


# In[119]:


df.shape


# In[120]:


df.head()


# In[121]:


latlong = pd.read_csv('latlong.csv')


# In[123]:


latlong


# In[124]:


latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')


# In[125]:


latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')


# In[126]:


latlong.head()


# In[127]:


new_df = df.merge(latlong, on='sector')


# In[145]:


new_df.columns


# In[129]:


group_df = new_df.groupby('sector').mean()[['price','price_per_sqft','built_up_area','latitude','longitude']]


# In[130]:


group_df


# In[131]:


fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                  mapbox_style="open-street-map",text=group_df.index)
fig.show()


# In[52]:


new_df.to_csv('data_viz1.csv',index=False)


# In[55]:


df1 = pd.read_csv('gurgaon_properties.csv')


# In[56]:


df1.head()


# In[132]:


wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]


# In[133]:


wordcloud_df.head()


# In[137]:


import ast
main = []
for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
    main.extend(item)


# In[138]:


main


# In[141]:


from wordcloud import WordCloud


# In[139]:


feature_text = ' '.join(main)


# In[143]:


import pickle
pickle.dump(feature_text, open('feature_text.pkl','wb'))


# In[140]:


feature_text


# In[142]:


plt.rcParams["font.family"] = "Arial"

wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='white', 
                      stopwords = set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size = 10).generate(feature_text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() # st.pyplot()


# In[100]:


data = dict(
    names=["A", "B", "C", "D", "E", "F"],
    parents=["", "", "", "A", "A", "C"],
    values=[10, 20, 30, 40, 50, 60],
)

fig = px.sunburst(
    df1,
    names='property_type',
    values='price_per_sqft',
    parents='bedRoom',
    title="Sample Sunburst Chart"
)
fig.show()


# In[144]:


fig = px.scatter(df, x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")

# Show the plot
fig.show()


# In[105]:


fig = px.pie(df, names='bedRoom', title='Total Bill Amount by Day')

# Show the plot
fig.show()


# In[109]:


temp_df = df[df['bedRoom'] <= 4]
# Create side-by-side boxplots of the total bill amounts by day
fig = px.box(temp_df, x='bedRoom', y='price', title='BHK Price Range')

# Show the plot
fig.show()


# In[111]:


sns.distplot(df[df['property_type'] == 'house']['price'])
sns.distplot(df[df['property_type'] == 'flat']['price'])


# In[150]:


new_df['sector'].unique().tolist().insert(0,'overall')


# In[ ]:




