"""
🎬 Movie Recommendation Web App
Flask backend serving the recommendation API
"""

from flask import Flask, render_template, jsonify, request
from movie_recommender import MovieRecommender
import os

app = Flask(__name__)

# Initialize the recommender
print("🎬 Loading Movie Recommendation Model...")
recommender = MovieRecommender(data_path="data")

# Load or build model
if os.path.exists("movie_model.pkl"):
    recommender.load_model()
else:
    recommender.load_data()
    recommender.preprocess_data()
    recommender.create_tags()
    recommender.build_model()
    recommender.save_model()

print("✅ Model ready!")


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint to get movie recommendations"""
    data = request.get_json()
    movie_title = data.get('movie', '')
    num_recs = data.get('count', 6)
    
    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400
    
    # Find movie
    movie_matches = recommender.movies_df[
        recommender.movies_df['title'].str.lower() == movie_title.lower()
    ]
    
    if len(movie_matches) == 0:
        # Try partial match
        movie_matches = recommender.movies_df[
            recommender.movies_df['title'].str.lower().str.contains(movie_title.lower())
        ]
        if len(movie_matches) == 0:
            return jsonify({
                'error': f"Movie '{movie_title}' not found",
                'suggestions': search_movies(movie_title)
            }), 404
    
    movie_index = movie_matches.index[0]
    movie_found = recommender.movies_df.loc[movie_index, 'title']
    
    # Get similarity scores
    distances = recommender.similarity_matrix[movie_index]
    
    # Get top similar movies
    movie_list = sorted(
        list(enumerate(distances)), 
        key=lambda x: x[1], 
        reverse=True
    )[1:num_recs + 1]
    
    recommendations = []
    for idx, score in movie_list:
        title = recommender.movies_df.iloc[idx]['title']
        recommendations.append({
            'title': title,
            'similarity': round(score * 100, 1)
        })
    
    return jsonify({
        'movie': movie_found,
        'recommendations': recommendations
    })


@app.route('/api/search', methods=['GET'])
def search():
    """Search for movies (for autocomplete)"""
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    
    matches = recommender.movies_df[
        recommender.movies_df['title'].str.lower().str.contains(query.lower())
    ]['title'].head(10).tolist()
    
    return jsonify(matches)


@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    """Get list of all movie titles"""
    movies = recommender.movies_df['title'].tolist()
    return jsonify(movies)


def search_movies(query):
    """Helper function to search movies"""
    matches = recommender.movies_df[
        recommender.movies_df['title'].str.lower().str.contains(query.lower())
    ]['title'].head(5).tolist()
    return matches


if __name__ == '__main__':
    app.run(debug=True, port=5000)
