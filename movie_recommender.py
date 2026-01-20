"""
🎬 Movie Recommendation System
==============================
A content-based movie recommendation engine using TMDB dataset.

HOW IT WORKS:
1. Load movie data (titles, genres, keywords, cast, crew, overview)
2. Combine all features into a single "tags" column
3. Convert text to numerical vectors using CountVectorizer
4. Calculate similarity between all movies using Cosine Similarity
5. For any movie, find the most similar movies and recommend them

Key Concepts:
- CountVectorizer: Converts text into a matrix of word counts
- Cosine Similarity: Measures angle between two vectors (0=different, 1=identical)
- Stemming: Reduces words to root form (loving, loved, loves → love)
"""

import pandas as pd
import numpy as np
import ast
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class MovieRecommender:
    """
    Content-Based Movie Recommendation System
    
    This class handles:
    - Loading and preprocessing movie data
    - Feature extraction (genres, keywords, cast, crew, overview)
    - Building the similarity matrix
    - Generating movie recommendations
    """
    
    def __init__(self, data_path="data"):
        """Initialize the recommender with path to data files."""
        self.data_path = data_path
        self.movies_df = None
        self.similarity_matrix = None
        self.stemmer = PorterStemmer()
        
    def load_data(self):
        """
        Step 1: Load the CSV files and merge them.
        
        We have two files:
        - tmdb_5000_movies.csv: Contains movie info (title, overview, genres, keywords)
        - tmdb_5000_credits.csv: Contains cast and crew info
        """
        print("\n📚 Loading dataset...")
        
        movies_path = os.path.join(self.data_path, "tmdb_5000_movies.csv")
        credits_path = os.path.join(self.data_path, "tmdb_5000_credits.csv")
        
        # Load both CSV files
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
        
        print(f"   Movies shape: {movies.shape}")
        print(f"   Credits shape: {credits.shape}")
        
        # Merge on 'title' column
        # This combines movie info with its cast/crew
        self.movies_df = movies.merge(credits, on='title')
        
        print(f"   Merged shape: {self.movies_df.shape}")
        print(f"\n📋 Columns available: {list(self.movies_df.columns)}")
        
        return self.movies_df
    
    def _convert_to_list(self, obj):
        """
        Helper function to convert JSON string to Python list.
        
        The CSV stores genres, keywords, cast as JSON strings like:
        '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        
        We extract just the names: ["Action", "Adventure"]
        """
        try:
            L = ast.literal_eval(obj)
            return [item['name'] for item in L]
        except:
            return []
    
    def _get_top3(self, obj):
        """Get only top 3 items (used for cast - we take top 3 actors)."""
        try:
            L = ast.literal_eval(obj)
            return [item['name'] for item in L[:3]]
        except:
            return []
    
    def _get_director(self, obj):
        """Extract director's name from crew data."""
        try:
            L = ast.literal_eval(obj)
            for item in L:
                if item['job'] == 'Director':
                    return [item['name']]
            return []
        except:
            return []
    
    def preprocess_data(self):
        """
        Step 2: Clean and preprocess the data.
        
        We keep only the columns we need:
        - movie_id, title: For identification
        - overview: Movie plot summary
        - genres: Action, Comedy, Drama, etc.
        - keywords: Theme keywords
        - cast: Top 3 actors
        - crew: Director name
        """
        print("\n🔧 Preprocessing data...")
        
        # Keep only necessary columns
        self.movies_df = self.movies_df[['movie_id', 'title', 'overview', 
                                          'genres', 'keywords', 'cast', 'crew']]
        
        # Drop rows with missing values
        self.movies_df = self.movies_df.dropna()
        
        print(f"   Rows after removing nulls: {len(self.movies_df)}")
        
        # Convert JSON strings to lists
        print("   Converting genres...")
        self.movies_df['genres'] = self.movies_df['genres'].apply(self._convert_to_list)
        
        print("   Converting keywords...")
        self.movies_df['keywords'] = self.movies_df['keywords'].apply(self._convert_to_list)
        
        print("   Extracting top 3 cast members...")
        self.movies_df['cast'] = self.movies_df['cast'].apply(self._get_top3)
        
        print("   Extracting director...")
        self.movies_df['crew'] = self.movies_df['crew'].apply(self._get_director)
        
        # Split overview into list of words
        self.movies_df['overview'] = self.movies_df['overview'].apply(lambda x: x.split())
        
        # Remove spaces from names (e.g., "Sam Worthington" → "SamWorthington")
        # This prevents "Sam" from matching with other "Sam"s
        self.movies_df['genres'] = self.movies_df['genres'].apply(
            lambda x: [i.replace(" ", "") for i in x])
        self.movies_df['keywords'] = self.movies_df['keywords'].apply(
            lambda x: [i.replace(" ", "") for i in x])
        self.movies_df['cast'] = self.movies_df['cast'].apply(
            lambda x: [i.replace(" ", "") for i in x])
        self.movies_df['crew'] = self.movies_df['crew'].apply(
            lambda x: [i.replace(" ", "") for i in x])
        
        print("   ✅ Preprocessing complete!")
        
        return self.movies_df
    
    def create_tags(self):
        """
        Step 3: Create a 'tags' column by combining all features.
        
        For each movie, we combine:
        - overview words
        - genres
        - keywords  
        - cast names
        - director name
        
        Into a single string like:
        "action adventure hero saves world ChristianBale ChristopherNolan"
        """
        print("\n🏷️  Creating tags column...")
        
        # Combine all features into one column
        self.movies_df['tags'] = (
            self.movies_df['overview'] + 
            self.movies_df['genres'] + 
            self.movies_df['keywords'] + 
            self.movies_df['cast'] + 
            self.movies_df['crew']
        )
        
        # Keep only necessary columns for the model
        self.movies_df = self.movies_df[['movie_id', 'title', 'tags']]
        
        # Convert list to lowercase string
        self.movies_df['tags'] = self.movies_df['tags'].apply(
            lambda x: " ".join(x).lower())
        
        print(f"\n📝 Example tags for '{self.movies_df.iloc[0]['title']}':")
        print(f"   {self.movies_df.iloc[0]['tags'][:200]}...")
        
        return self.movies_df
    
    def _stem_text(self, text):
        """
        Apply stemming to reduce words to their root form.
        
        Examples:
        - "loving", "loved", "loves" → "love"
        - "dancing", "danced" → "danc"
        
        This helps match similar words together.
        """
        words = text.split()
        stemmed = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed)
    
    def build_model(self):
        """
        Step 4: Build the recommendation model.
        
        Process:
        1. Apply stemming to all tags
        2. Convert text to numerical vectors using CountVectorizer
        3. Calculate cosine similarity between all movie pairs
        
        CountVectorizer creates a matrix where:
        - Each row is a movie
        - Each column is a unique word
        - Values are word counts
        
        Cosine Similarity creates a 4800x4800 matrix where:
        - similarity[i][j] = how similar movie i is to movie j
        """
        print("\n🤖 Building recommendation model...")
        
        # Apply stemming
        print("   Applying stemming to tags...")
        self.movies_df['tags'] = self.movies_df['tags'].apply(self._stem_text)
        
        # Convert text to vectors
        # max_features=5000: Keep only top 5000 most common words
        # stop_words='english': Remove common words like "the", "is", "at"
        print("   Converting text to vectors (CountVectorizer)...")
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(self.movies_df['tags']).toarray()
        
        print(f"   Vector shape: {vectors.shape}")
        print(f"   (Each movie is represented by {vectors.shape[1]} features)")
        
        # Calculate similarity matrix
        print("   Calculating cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(vectors)
        
        print(f"   Similarity matrix shape: {self.similarity_matrix.shape}")
        print("   ✅ Model built successfully!")
        
        return self.similarity_matrix
    
    def recommend(self, movie_title, num_recommendations=5):
        """
        Step 5: Get movie recommendations!
        
        Process:
        1. Find the index of the input movie
        2. Get similarity scores with all other movies
        3. Sort by similarity (highest first)
        4. Return top N recommendations
        
        Args:
            movie_title: Name of the movie to get recommendations for
            num_recommendations: Number of movies to recommend (default 5)
        
        Returns:
            List of recommended movie titles
        """
        # Find movie index (case-insensitive search)
        movie_matches = self.movies_df[
            self.movies_df['title'].str.lower() == movie_title.lower()
        ]
        
        if len(movie_matches) == 0:
            # Try partial match
            movie_matches = self.movies_df[
                self.movies_df['title'].str.lower().str.contains(movie_title.lower())
            ]
            if len(movie_matches) == 0:
                print(f"❌ Movie '{movie_title}' not found in database!")
                return self.search_movies(movie_title)
            
        movie_index = movie_matches.index[0]
        movie_found = self.movies_df.loc[movie_index, 'title']
        
        # Get similarity scores for this movie with all others
        distances = self.similarity_matrix[movie_index]
        
        # Get indices of top similar movies (exclude the movie itself)
        # enumerate creates (index, similarity) pairs
        # sorted by similarity descending
        movie_list = sorted(
            list(enumerate(distances)), 
            key=lambda x: x[1], 
            reverse=True
        )[1:num_recommendations + 1]  # Skip first (itself)
        
        # Get movie titles
        recommendations = []
        print(f"\n🎬 Movies similar to '{movie_found}':\n")
        print("-" * 50)
        
        for i, (idx, score) in enumerate(movie_list, 1):
            title = self.movies_df.iloc[idx]['title']
            recommendations.append(title)
            print(f"   {i}. {title}")
            print(f"      Similarity Score: {score:.2%}")
        
        print("-" * 50)
        return recommendations
    
    def search_movies(self, query):
        """Search for movies containing the query string."""
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query.lower())
        ]['title'].head(10).tolist()
        
        if matches:
            print(f"\n🔍 Did you mean one of these?")
            for i, title in enumerate(matches, 1):
                print(f"   {i}. {title}")
        return matches
    
    def save_model(self, filename="movie_model.pkl"):
        """Save the trained model to disk for faster loading later."""
        print(f"\n💾 Saving model to {filename}...")
        
        model_data = {
            'movies_df': self.movies_df,
            'similarity_matrix': self.similarity_matrix
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("   ✅ Model saved successfully!")
    
    def load_model(self, filename="movie_model.pkl"):
        """Load a pre-trained model from disk."""
        print(f"\n📂 Loading model from {filename}...")
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.movies_df = model_data['movies_df']
        self.similarity_matrix = model_data['similarity_matrix']
        
        print("   ✅ Model loaded successfully!")
        print(f"   Total movies: {len(self.movies_df)}")


def main():
    """Main function to run the recommendation system."""
    
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    recommender = MovieRecommender(data_path="data")
    
    # Check if pre-trained model exists
    if os.path.exists("movie_model.pkl"):
        recommender.load_model()
    else:
        # Build model from scratch
        recommender.load_data()
        recommender.preprocess_data()
        recommender.create_tags()
        recommender.build_model()
        recommender.save_model()
    
    # Interactive recommendation loop
    print("\n" + "=" * 60)
    print("🎯 READY TO RECOMMEND!")
    print("=" * 60)
    print("\nEnter a movie name to get recommendations.")
    print("Type 'quit' to exit.\n")
    
    while True:
        movie = input("🎬 Enter movie name: ").strip()
        
        if movie.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using Movie Recommender! Goodbye!")
            break
        
        if movie:
            recommender.recommend(movie)
        print()


if __name__ == "__main__":
    main()
