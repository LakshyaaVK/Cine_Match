/**
 * CineMatch - Movie Recommendation System
 * Frontend JavaScript
 */

// DOM Elements
const movieInput = document.getElementById('movieInput');
const searchBtn = document.getElementById('searchBtn');
const autocomplete = document.getElementById('autocomplete');
const resultsSection = document.getElementById('results');
const selectedMovieSpan = document.getElementById('selectedMovie');
const movieGrid = document.getElementById('movieGrid');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const errorText = document.getElementById('errorText');
const errorSuggestions = document.getElementById('errorSuggestions');
const suggestionChips = document.querySelectorAll('.suggestion-chip');

// Movie icons for variety
const movieIcons = ['🎬', '🎥', '🎞️', '🍿', '🎭', '🌟', '✨', '🎪', '🎨', '🎯'];

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Show loading state
function showLoading() {
    loading.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorDiv.classList.add('hidden');
}

// Hide loading state
function hideLoading() {
    loading.classList.add('hidden');
}

// Show error message
function showError(message, suggestions = []) {
    hideLoading();
    resultsSection.classList.add('hidden');
    errorDiv.classList.remove('hidden');
    errorText.textContent = message;

    // Add suggestion buttons
    errorSuggestions.innerHTML = '';
    suggestions.forEach(movie => {
        const btn = document.createElement('button');
        btn.className = 'error-suggestion-btn';
        btn.textContent = movie;
        btn.addEventListener('click', () => {
            movieInput.value = movie;
            getRecommendations(movie);
        });
        errorSuggestions.appendChild(btn);
    });
}

// Show results
function showResults(movie, recommendations) {
    hideLoading();
    errorDiv.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    selectedMovieSpan.textContent = movie;

    // Clear and populate grid
    movieGrid.innerHTML = '';

    recommendations.forEach((rec, index) => {
        const card = createMovieCard(rec, index);
        movieGrid.appendChild(card);
    });
}

// Create movie card element
function createMovieCard(movie, index) {
    const card = document.createElement('div');
    card.className = 'movie-card';

    const icon = movieIcons[index % movieIcons.length];
    const similarity = movie.similarity;

    card.innerHTML = `
        <div class="movie-rank">#${index + 1}</div>
        <div class="movie-icon">${icon}</div>
        <h3 class="movie-title">${movie.title}</h3>
        <div class="similarity-container">
            <div class="similarity-label">
                <span>Match Score</span>
                <span class="similarity-value">${similarity}%</span>
            </div>
            <div class="similarity-bar">
                <div class="similarity-fill" style="width: 0%"></div>
            </div>
        </div>
    `;

    // Animate similarity bar after a short delay
    setTimeout(() => {
        const fill = card.querySelector('.similarity-fill');
        fill.style.width = `${similarity}%`;
    }, 100 + (index * 100));

    return card;
}

// Get recommendations from API
async function getRecommendations(movieTitle) {
    if (!movieTitle.trim()) {
        alert('Please enter a movie name');
        return;
    }

    showLoading();
    hideAutocomplete();

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                movie: movieTitle,
                count: 6
            })
        });

        const data = await response.json();

        if (response.ok) {
            showResults(data.movie, data.recommendations);
        } else {
            showError(data.error, data.suggestions || []);
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the server. Please try again.');
    }
}

// Search movies for autocomplete
async function searchMovies(query) {
    if (query.length < 2) {
        hideAutocomplete();
        return;
    }

    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const movies = await response.json();

        if (movies.length > 0) {
            showAutocomplete(movies);
        } else {
            hideAutocomplete();
        }
    } catch (error) {
        console.error('Search error:', error);
    }
}

// Show autocomplete dropdown
function showAutocomplete(movies) {
    autocomplete.innerHTML = '';

    movies.forEach(movie => {
        const item = document.createElement('div');
        item.className = 'autocomplete-item';
        item.textContent = movie;
        item.addEventListener('click', () => {
            movieInput.value = movie;
            hideAutocomplete();
            getRecommendations(movie);
        });
        autocomplete.appendChild(item);
    });

    autocomplete.classList.add('show');
}

// Hide autocomplete dropdown
function hideAutocomplete() {
    autocomplete.classList.remove('show');
}

// Debounced search
const debouncedSearch = debounce(searchMovies, 300);

// Event Listeners
movieInput.addEventListener('input', (e) => {
    debouncedSearch(e.target.value);
});

movieInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        hideAutocomplete();
        getRecommendations(movieInput.value);
    }
});

searchBtn.addEventListener('click', () => {
    getRecommendations(movieInput.value);
});

// Suggestion chips
suggestionChips.forEach(chip => {
    chip.addEventListener('click', () => {
        const movie = chip.dataset.movie;
        movieInput.value = movie;
        getRecommendations(movie);
    });
});

// Hide autocomplete when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-container')) {
        hideAutocomplete();
    }
});

// Initialize - Focus on input
movieInput.focus();

console.log('🎬 CineMatch loaded successfully!');
