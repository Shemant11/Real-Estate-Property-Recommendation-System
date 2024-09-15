
# Real Estate Price Prediction and Recommendation System

## Project Overview

This project leverages data science techniques to provide insights, predictions, and personalized recommendations for real estate properties. The goal is to predict real estate prices accurately and provide personalized property recommendations based on user preferences such as location, facilities, and price. The project utilizes **machine learning models** for price estimation and a **recommendation engine** to enhance user experience. It integrates various steps including data scraping, cleaning, feature engineering, exploratory data analysis, model development, and deployment via a web-based interface using **Streamlit**.


---

## Key Features

- **Data Collection & Cleaning**: Scraped property data from 99acres using Python, BeautifulSoup. Cleaned and merged data using Pandas and NumPy, handling missing values, outliers, and inconsistencies.
  
- **Feature Engineering**: Created new features such as luxury scores, furniture details, and area type specifications to improve property price predictions and recommendations.
  
- **Exploratory Data Analysis (EDA)**: Conducted EDA using Pandas Profiling, Matplotlib, and Seaborn to uncover key patterns in property data. Automated EDA reports provided insights into data distribution, missing values, and feature interactions.
  
- **Model Development**: Built and compared several machine learning models including Linear Regression, Random Forest, Gradient Boosting, LASSO, Ridge, ElasticNet, and more, to predict property prices.
  
- **Recommendation System**: Designed a personalized recommendation system using cosine similarity and Scikit-learn to recommend properties based on user preferences such as facilities, price, and location.
  
- **Deployment**: Deployed the model and recommendation system using **Streamlit**, providing a user-friendly interface for real-time price predictions and personalized recommendations.
  
- **Custom Analytics Module**: Built an analytics module using Seaborn, Plotly, and Folium to visualize real estate data through geographical maps, scatter plots, pie charts, and word clouds, offering comprehensive insights into the real estate market.
---
## Tech Stack

- **Languages & Libraries**: Python, Pandas, NumPy, BeautifulSoup, Scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly, Streamlit
- **Machine Learning Models**: Linear Regression, Random Forest, Gradient Boosting, LASSO, Ridge, ElasticNet, K-Nearest Neighbors, Decision Tree
- **Deployment**: Streamlit for web app development
- **Python**: Core language used for scripting, data processing, and modeling.
- **BeautifulSoup**: Web scraping tools used to collect data from real estate platforms like 99acres.
- **Pandas & NumPy**: For data cleaning, merging, and preprocessing.
- **Pandas Profiling**: Automated exploratory data analysis.
- **Scikit-learn & XGBoost**: Used to build and evaluate various machine learning models.
- **Streamlit**: For deploying the application and building a web-based interface.
- **Seaborn, Matplotlib, Plotly**: For data visualization and insights.
---

## Usage

- Access property price predictions and personalized recommendations through the Streamlit interface.
- Use the analytics module to explore the real estate market with geographical maps, scatter plots, pie charts, and word clouds.

---


## Data Collection

- Web scraping is used to gather real estate property data from **99acres** and similar platforms using **BeautifulSoup**.
- Data is cleaned, processed, and merged to ensure consistency and handle missing values, outliers, and inconsistencies.

## Feature Engineering

- New features such as **luxury score**, **furniture details**, and **area type** were created using **Pandas** and **FeatureTools** to improve prediction performance.

## Exploratory Data Analysis (EDA)

- Conducted **EDA** using **Pandas Profiling**, **Matplotlib**, and **Seaborn** to uncover trends and patterns in property data.
- Automated **EDA** reports generated for comprehensive insights into variable distributions, missing data, and correlations.

## Model Development

- Implemented and compared multiple machine learning models using **Scikit-learn** and **XGBoost**, including:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - LASSO, Ridge, and ElasticNet
- Evaluated model performance using metrics like **R²**, **MAE**, and **RMSE**.

## Recommendation System

- A **recommendation engine** built using **cosine similarity** and **Scikit-learn** to suggest properties based on user preferences (facilities, location, price).
- Integrated into the **Streamlit** interface for ease of use and accessibility.

## Data Visualization

- Created an analytics module with **Seaborn**, **Plotly**, to visualize geographical data, property trends, and word clouds.
  
## Deployment

- The **Streamlit** application provides an intuitive interface for real-time property price estimation and personalized recommendations.

## Future Enhancements

- Integration of more advanced models like **deep learning** for further accuracy improvements.
- Expansion of recommendation system to include more complex user preferences and factors.
- Implementation of live data updates from property listing platforms.

## Screenshots

![Property Price Prediction](screenshots/prediction.png)
*Price prediction interface*

![Recommendation System](screenshots/recommendation.png)
*Recommendation system interface*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
