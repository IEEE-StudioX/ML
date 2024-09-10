import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
@st.cache  # Cache the data to avoid reloading on every interaction
def load_data():
    ratings = pd.read_csv('https://raw.githubusercontent.com/IEEE-StudioX/ML/main/API/ratings.csv')
    movies = pd.read_csv("https://raw.githubusercontent.com/IEEE-StudioX/ML/main/API/movies.csv")
    return ratings, movies

ratings, movies = load_data()

# Preprocess the data and prepare the user-item matrix
@st.cache
def prepare_data(ratings, movies):
    merged_data = pd.merge(ratings, movies, on='movieId', how='inner')
    user_item_matrix = merged_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    return user_item_matrix

user_item_matrix = prepare_data(ratings, movies)

# KNN Model training
@st.cache
def train_knn_model(user_item_matrix):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(csr_matrix(user_item_matrix.values))
    return model_knn

model_knn = train_knn_model(user_item_matrix)

# Function to get movie recommendations
def get_movie_recommendations(movie_title, user_item_matrix, model_knn, n_recommendations=5):
    movie_idx = user_item_matrix.columns.get_loc(movie_title)
    distances, indices = model_knn.kneighbors(user_item_matrix.iloc[:, movie_idx].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommendations.append(user_item_matrix.columns[indices.flatten()[i]])
    
    return recommendations

# Streamlit user interface
st.title("KNN Item-Based Movie Recommendation System")

# User input for movie title
movie_title = st.selectbox('Select a movie you like:', user_item_matrix.columns)

# Number of recommendations input
n_recommendations = st.slider('Number of recommendations', 1, 20, 5)

# Recommend button
if st.button('Get Recommendations'):
    recommendations = get_movie_recommendations(movie_title, user_item_matrix, model_knn, n_recommendations)
    
    st.write(f"Movies recommended based on your interest in {movie_title}:")
    for idx, rec in enumerate(recommendations):
        st.write(f"{idx+1}. {rec}")