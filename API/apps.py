import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import requests

# Load data and prepare user-item matrix
ratings = pd.read_csv("https://github.com/IEEE-StudioX/ML/blob/main/API/ratings.csv")
movies = pd.read_csv("https://github.com/IEEE-StudioX/ML/blob/main/API/movies.csv")
merged_data = pd.merge(ratings, movies, on='movieId', how='inner')

# Convert user-item matrix to a sparse matrix format
user_item_matrix = merged_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Compute user similarity
user_similarity = cosine_similarity(user_item_matrix_sparse)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to recommend movies based on collaborative filtering
def recommend_movies(user_id, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(10).index
    similar_user_ratings = user_item_matrix.loc[similar_users]
    weighted_ratings = similar_user_ratings.mean().sort_values(ascending=False)
    already_rated = user_item_matrix.loc[user_id]
    recommendations = weighted_ratings[~already_rated.index.isin(already_rated[already_rated > 0].index)]
    return recommendations.head(num_recommendations)

# Streamlit interface
st.title("User-Based Movie Recommendation System")

# User input for ID
user_id = st.number_input("Enter your User ID:", min_value=int(user_item_matrix.index.min()), max_value=int(user_item_matrix.index.max()))

# Number of recommendations input
num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

# Button to generate recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_id=user_id, num_recommendations=num_recommendations)
    st.write("### Movie Recommendations:")
    for idx, (movie, rating) in enumerate(recommendations.items(), 1):
        st.write(f"{idx}. {movie} (Predicted Rating: {rating:.2f})")