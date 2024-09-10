import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
@st.cache_data  # Caches data-related operations like loading CSV files
def load_data():
    ratings = pd.read_csv('https://raw.githubusercontent.com/IEEE-StudioX/ML/main/API/ratings.csv')
    movies = pd.read_csv("https://raw.githubusercontent.com/IEEE-StudioX/ML/main/API/movies.csv")
    return ratings, movies

# Cache the preparation of the user-item matrix
@st.cache_data
def prepare_data(ratings, movies):
    merged_data = pd.merge(ratings, movies, on='movieId', how='inner')
    user_item_matrix = merged_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    return user_item_matrix

# Cache the model training to avoid retraining
@st.cache_resource
def train_knn_model(item_item_matrix):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(csr_matrix(item_item_matrix.values))  # Train the KNN on the transposed matrix
    return model_knn

# Movie recommendation function
def get_movie_recommendations(movie_title, item_item_matrix, model_knn, n_recommendations=5):
    # Get the index of the movie in the item-item matrix (transposed user-item matrix)
    movie_idx = item_item_matrix.index.get_loc(movie_title)
    
    # Ensure you are using the correct shape of the matrix
    movie_vector = item_item_matrix.iloc[movie_idx, :].values.reshape(1, -1)
    
    # Perform KNN search to find similar movies
    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=n_recommendations+1)
    
    # Get the recommended movies based on nearest neighbors
    recommendations = [item_item_matrix.index[indices.flatten()[i]] for i in range(1, len(distances.flatten()))]
    
    return recommendations

# Streamlit app
def main():
    st.title("KNN Item-Based Movie Recommendation System")

    # Load data
    ratings, movies = load_data()
    user_item_matrix = prepare_data(ratings, movies)
    
    # Transpose the user-item matrix to create an item-item matrix (items as rows, users as columns)
    item_item_matrix = user_item_matrix.T
    
    # Train the KNN model on the item-item matrix
    model_knn = train_knn_model(item_item_matrix)

    # Select movie title
    movie_title = st.selectbox('Select a movie you like:', item_item_matrix.index)

    # Select number of recommendations
    n_recommendations = st.slider('Number of recommendations', 1, 20, 5)

    # Get recommendations
    if st.button('Get Recommendations'):
        recommendations = get_movie_recommendations(movie_title, item_item_matrix, model_knn, n_recommendations)
        st.write(f"Movies recommended based on your interest in {movie_title}:")
        for idx, rec in enumerate(recommendations):
            st.write(f"{idx+1}. {rec}")

# Ensure the main function runs when the script is executed directly
if __name__ == '__main__':
    main()