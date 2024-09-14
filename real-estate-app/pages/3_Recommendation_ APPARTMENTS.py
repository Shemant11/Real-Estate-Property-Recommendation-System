"""
@author: HEMANT
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recommend Appartments")

# Load the location distance dataframe using pd.read_pickle instead of pickle.load
try:
    location_df = pd.read_pickle(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\datasets\location_distance.pkl")
except Exception as e:
    st.error(f"Error loading location data: {e}")

# Load cosine similarity matrices with proper paths
try:
    cosine_sim1 = pickle.load(open(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\datasets\cosine_sim1.pkl", 'rb'))
    cosine_sim2 = pickle.load(open(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\datasets\cosine_sim2.pkl", 'rb'))
    cosine_sim3 = pickle.load(open(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\datasets\cosine_sim3.pkl", 'rb'))
except Exception as e:
    st.error(f"Error loading cosine similarity data: {e}")

# Function to recommend properties with similarity scores
def recommend_properties_with_scores(property_name, top_n=5):
    try:
        # Combined cosine similarity matrix
        cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3

        # Get the similarity scores for the property using its name as the index
        sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))

        # Sort properties based on the similarity scores
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices and scores of the top_n most similar properties
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

        # Retrieve the names of the top properties using the indices
        top_properties = location_df.index[top_indices].tolist()

        # Create a dataframe with the results
        recommendations_df = pd.DataFrame({
            'PropertyName': top_properties,
            'SimilarityScore': top_scores
        })

        return recommendations_df
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Test the recommender function using a property name
recommend_properties_with_scores('DLF The Camellias')


st.title('Select Location and Radius')

# Check if location_df is loaded properly before using it
if 'location_df' in locals():
    selected_location = st.selectbox('Location', sorted(location_df.columns.to_list()))

    radius = st.number_input('Radius in Kms')

    if st.button('Search'):
        try:
            result_ser = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()

            for key, value in result_ser.items():
                st.text(f"{key} {round(value / 1000, 2)} kms")
        except Exception as e:
            st.error(f"Error searching locations: {e}")

st.title('Recommend Apartments')

# Check if location_df is loaded properly before using it
if 'location_df' in locals():
    selected_apartment = st.selectbox('Select an apartment', sorted(location_df.index.to_list()))

    if st.button('Recommend'):
        recommendation_df = recommend_properties_with_scores(selected_apartment)

        if not recommendation_df.empty:
            st.dataframe(recommendation_df)
        else:
            st.error("No recommendations found.")
