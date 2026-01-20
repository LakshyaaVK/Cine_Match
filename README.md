# 🎬 CineMatch - Movie Recommendation System

A content-based movie recommendation system built with Python that suggests similar movies based on genres, cast, crew, and keywords.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange.svg)



##live preview(gradio) https://huggingface.co/spaces/slaygen/CineMatch



## 🛠️ How It Works

1. **Feature Extraction**: Extracts genres, keywords, top 3 cast members, director, and overview from each movie
2. **Tag Creation**: Combines all features into a single text representation
3. **Vectorization**: Converts text to numerical vectors using CountVectorizer
4. **Similarity Calculation**: Uses Cosine Similarity to find movies with similar characteristics
5. **Recommendation**: Returns the most similar movies based on the similarity scores

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation.git
cd movie-recommendation
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
```bash
python download_data.py
```

### 5. Run the web app
```bash
python app.py
```

### 6. Open in browser
Navigate to: http://127.0.0.1:5000

## 📁 Project Structure

```
movie_reccomendation/
├── app.py                    # Flask web server
├── movie_recommender.py      # Core recommendation engine
├── download_data.py          # Dataset downloader
├── requirements.txt          # Dependencies
├── templates/
│   └── index.html            # Web frontend
├── static/
│   ├── css/style.css         # Styling
│   └── js/app.js             # Frontend JavaScript
└── data/                     # Dataset (auto-downloaded)
```

## 🧠 Algorithm Details

### CountVectorizer
Converts movie tags (text) into a matrix of word counts. Each movie becomes a vector of 5000 dimensions.

### Cosine Similarity
Measures the angle between two movie vectors:
- **1.0** = Identical movies
- **0.0** = Completely different

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

## 📊 Dataset

This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.

## 🎨 Screenshots

The app features a beautiful dark theme with:
- Animated gradient background
- Glassmorphism cards
- Smooth animations
- Responsive design

## 📝 License

MIT License - feel free to use this project for learning and personal projects!

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) for the movie dataset
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [Flask](https://flask.palletsprojects.com/) for the web framework
