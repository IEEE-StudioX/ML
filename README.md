# Recommendation System Project

This repository contains the implementation of a recommendation system using different approaches, such as Item-based and User-based collaborative filtering, K-Nearest Neighbors (KNN), and cosine similarity.

## Project Structure

- **API Deployment with Streamlit**: Deployed a recommendation system using Streamlit for easy interaction and API integration.
  - **`streamlit_knn.py`**: Script for deploying the KNN recommendation model using Streamlit.

- **Datasets Used**:
  - The datasets are provided in the repository for the recommendation system tasks.
  
- **Cosine Similarity Explanation**:
  - **`cosine_similarity.html`**: Explains cosine similarity and how it is applied in recommendation systems.

- **Implementation Notebooks**:
  - **`KNN_Item_based_recommendation_system.ipynb`**: Contains the implementation of the KNN-based item recommendation system.
  - **`User_Base.ipynb`**: Implements a user-based recommendation system.
  - **`item-based-recommender-system (1).ipynb`**: Final version of the item-based recommender system implemented with the cosine similarity.
  
- **Helper Scripts**:
  - **`libraries.py`**: Contains all the libraries used in the project.
  - **`user-item_matrix.py`**: Script to create and manipulate the user-item matrix.

## Key Concepts

### 1. K-Nearest Neighbors (KNN) 
KNN is used to find the most similar items or users based on a defined similarity metric.

### 2. Cosine Similarity
Cosine similarity is a metric used to measure the cosine of the angle between two non-zero vectors. It's often used in item-based recommendation systems to determine the similarity between two items.

## How to Use

1. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt


