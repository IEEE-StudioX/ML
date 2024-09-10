from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import requests  # External API
import os  # For accessing environment variables securely

# Load data and prepare user-item matrix
ratings = pd.read_csv("C:\\Users\\ACER\\Downloads\\Telegram Desktop\\ratings.csv")
movies = pd.read_csv("C:\\Users\\ACER\\Downloads\\Telegram Desktop\\movies.csv")
merged_data = pd.merge(ratings, movies, on='movieId', how='inner')

# Convert user-item matrix to a sparse matrix format
user_item_matrix = merged_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Compute user similarity
user_similarity = cosine_similarity(user_item_matrix_sparse)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Flask application
app = Flask(__name__)

# Movie Recommendation Function (based on collaborative filtering)
def recommend_movies(user_id, user_item_matrix, user_similarity_df, num_recommendations=5, top_n_similar_users=100):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(top_n_similar_users).index
    similar_user_ratings = user_item_matrix.loc[similar_users]
    weighted_ratings = similar_user_ratings.mean().sort_values(ascending=False)
    already_rated = user_item_matrix.loc[user_id]
    recommendations = weighted_ratings[~already_rated.index.isin(already_rated[already_rated > 0].index)]
    recommendations_df = pd.DataFrame(recommendations).reset_index()
    recommendations_df.columns = ['Movie Title', 'Predicted Rating']
    return recommendations_df.head(num_recommendations)

# TMDb API function to get additional recommendations
def external_api_recommendations(movie_title, api_key):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        response = requests.get(url)
        response_data = response.json()
        
        # Get the movie's id for further recommendations
        if response_data['results']:
            movie_id = response_data['results'][0]['id']
            recommendation_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={api_key}"
            recommendations_response = requests.get(recommendation_url)
            recommendations_data = recommendations_response.json()
            return [movie['title'] for movie in recommendations_data['results'][:5]]  # Return top 5 recommendations
        else:
            return []
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        return []

# Home page to enter user_id
@app.route('/')
def home():
    return '''
    <h1>Welcome to the Movie Recommendation System</h1>
    <form action="/recommend" method="get">
        Enter your user ID: <input type="text" name="user_id">
        <input type="submit" value="Get Recommendations">
    </form>
    '''

# API route to show recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    num_recommendations = int(request.args.get('num_recommendations', 5))
    
    # Call the recommendation function (collaborative filtering)
    internal_recommendations = recommend_movies(user_id=user_id, user_item_matrix=user_item_matrix, user_similarity_df=user_similarity_df, num_recommendations=num_recommendations)
    
    # Use an external API (e.g., TMDb) for more recommendations based on the top movie
    top_movie = internal_recommendations.iloc[0]['Movie Title']
    tmdb_api_key = os.getenv("TMDB_API_KEY")  # Make sure to set your API key as an environment variable
    external_recommendations = external_api_recommendations(top_movie, tmdb_api_key)
    
    # Combine both sets of recommendations into one response
    combined_recommendations = {
        'internal_recommendations': internal_recommendations.to_dict(orient='records'),
        'external_recommendations': external_recommendations
    }
    
    # Show the recommendations as HTML
    html_response = '<h1>Movie Recommendations</h1>'
    html_response += '<h2>Internal Recommendations:</h2><ul>'
    for rec in combined_recommendations['internal_recommendations']:
        html_response += f"<li>{rec['Movie Title']} - Predicted Rating: {rec['Predicted Rating']}</li>"
    html_response += '</ul>'
    
    html_response += '<h2>External Recommendations:</h2><ul>'
    for rec in combined_recommendations['external_recommendations']:
        html_response += f'<li>{rec}</li>'
    html_response += '</ul>'
    
    return html_response

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
