# Recommendation System Project

This repository contains the implementation of a recommendation system using different approaches, such as Item-based and User-based collaborative filtering, K-Nearest Neighbors (KNN), and cosine similarity.

## Note on Data Usage

We intended to use data coming directly from the app itself. However, since the app is not yet complete, we used a different dataset as a simulation of the data we will eventually extract from the app to test our code. 
The data we needed included: `user_id`, `item_id`, and `rating`.


## Project Structure

- **API Deployment with Streamlit**: Deployed a recommendation system using Streamlit and Flask for easy interaction and API integration.
  - **`APPknn.py`** :  Script for deploying the KNN Item_Based recommendation model as an API USING Streamlit.
  - **`apps.py`**   : Script for deploying the Cosine Similarity User_based recommendation model as an API USING Streamlit.
  - **`app.py`**    : Script for deploying the Cosine Similarity User_based recommendation model as an API USING Flask.

- **Datasets Used**:
  - The datasets are provided in the repository for the recommendation system tasks.
  
- **Cosine Similarity Explanation**:
  - **`cosine_similarity.html`**: Explains cosine similarity and how it is applied in recommendation systems.

- **Implementation Notebooks**:
  - **`KNN_Item_based_recommendation_system.ipynb`**: Contains the implementation of the KNN-based item recommendation system.
  - **`User_Base.ipynb`**: Implements a user-based recommendation system implemented by the cosine similarity..
  - **`item-based-recommender-system (1).ipynb`**: Final version of the item-based recommender system implemented by the cosine similarity.
  
- **Helper Scripts**:
  - **`libraries.py`**: Contains all the libraries used in the project.
  - **`user-item_matrix.py`**: Script to create and manipulate the user-item matrix.
  - **`requirements.txt`**: Lists all the required libraries and dependencies.
  - **`runtime.txt`**: Specifies the Python runtime environment.

## Key Concepts

### 1. K-Nearest Neighbors (KNN) 
KNN is used to find the most similar items or users based on a defined similarity metric.

### 2. Cosine Similarity
Cosine similarity is a metric used to measure the cosine of the angle between two non-zero vectors.

## How to Use

1. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt

2. ** to show recomendation system for user-item base
   - knn item-base : https://knnitembase.streamlit.app/
   - user base : https://userbaserecoomendation.streamlit.app/ 


