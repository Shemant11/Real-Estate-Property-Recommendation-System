"""
@author: HEMANT
"""

import streamlit as st
from PIL import Image

# Set the page configuration
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Main title
st.title("🏠 Real Estate Analytics Dashboard")

# Load an image for the header
header_image = Image.open(r"C:\Users\HEMANT\Downloads\Real Estate Property Recommendation System\real-estate-app\real_estate_header.jpg")  # Replace with your image file path

# Add a header image
st.image(header_image, use_column_width=True)



# Subtitle with a brief introduction
st.subheader("Analyze trends, make data-driven decisions, and find the perfect property.")

# Add some text about the app
st.markdown("""
Welcome to the **Real Estate Analytics Dashboard**, your one-stop platform for analyzing 
property trends, comparing market values, and making informed decisions. Explore the data, 
visualize insights, and stay ahead in the competitive real estate market.
""")

# Create sections on the page
st.header("Key Features")
st.markdown("""
- 📊 **Data-Driven Insights**: Analyze the latest property trends with real-time data.
- 📍 **Location-Based Analysis**: Compare properties across different areas.
- 💡 **Customizable Visualizations**: Tailor your analysis with a variety of charts and graphs.
- 🔍 **Market Comparisons**: Evaluate properties side by side and make informed choices.

""")

# Create a "Get Started" button
if st.button("Get Started"):
    st.write("Start analyzing properties now!")


# Footer section
st.markdown("""
    <hr>
    <p style='text-align: center;'>
    <b>Contact Us:</b> hemantkumar76910@gmail.com | <b>Phone:</b> +91 9521031040
    </p>
    """, unsafe_allow_html=True)
