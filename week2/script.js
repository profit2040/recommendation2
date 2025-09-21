// Initialize the application when the window loads
window.onload = async function() {
  try {
    const resultElement = document.getElementById('result');
    resultElement.textContent = "Loading movie data...";
    resultElement.className = 'loading';

    await loadData();

    populateMoviesDropdown();
    resultElement.textContent = "Data loaded. Please select a movie (Cosine Similarity).";
    resultElement.className = 'success';
  } catch (error) {
    console.error('Initialization error:', error);
    // data.js уже показал пользователю ошибку
  }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
  const selectElement = document.getElementById('movie-select');

  // Clear existing options except the first placeholder
  while (selectElement.options.length > 1) {
    selectElement.remove(1);
  }

  const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

  sortedMovies.forEach(movie => {
    const option = document.createElement('option');
    option.value = movie.id;
    option.textContent = movie.title;
    selectElement.appendChild(option);
  });
}

// Render recommendation cards
function renderRecommendations(likedMovie, recommendations) {
  const grid = document.getElementById('recommendations');
  if (!grid) return; // fallback: ничего не рендерим, если элемента нет

  // Очистить старые карточки
  grid.innerHTML = '';

  recommendations.forEach(rec => {
    const card = document.createElement('div');
    card.className = 'card';

    const title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = rec.title;

    const meta = document.createElement('div');
    meta.className = 'meta';

    const score = document.createElement('div');
    score.className = 'score';
    // Показываем косинус в процентах с одним знаком
    score.textContent = `Cosine: ${(rec.score * 100).toFixed(1)}%`;

    meta.appendChild(score);

    const chipsWrap = document.createElement('div');
    chipsWrap.className = 'chips';
    (rec.genres || []).forEach(g => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = g;
      chipsWrap.appendChild(chip);
    });

    card.appendChild(title);
    card.appendChild(meta);
    card.appendChild(chipsWrap);

    grid.appendChild(card);
  });
}

// Main recommendation function
function getRecommendations() {
  const resultElement = document.getElementById('result');

  try {
    // Step 1: Get user input
    const selectElement = document.getElementById('movie-select');
    const selectedMovieId = parseInt(selectElement.value);

    if (isNaN(selectedMovieId)) {
      resultElement.textContent = "Please select a movie first.";
      resultElement.className = 'error';
      renderRecommendations(null, []);
      return;
    }

    // Step 2: Find the liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
      resultElement.textContent = "Error: Selected movie not found in database.";
      resultElement.className = 'error';
      renderRecommendations(null, []);
      return;
    }

    // Show loading message while processing
    resultElement.textContent = "Calculating recommendations (Cosine Similarity)...";
    resultElement.className = 'loading';

    // Allow UI to update before computation
    setTimeout(() => {
      try {
        // Step 3: Prepare for similarity calculation
        const likedGenres = new Set(likedMovie.genres);
        const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);

        // Step 4: Calculate Cosine similarity scores (binary features)
        const scoredMovies = candidateMovies.map(candidate => {
          const candidateGenres = new Set(candidate.genres);

          // Intersection size = dot product for binary vectors
          const intersectionSize = [...likedGenres].filter(genre => candidateGenres.has(genre)).length;

          // Denominator: sqrt(|A| * |B|)
          const denom = Math.sqrt(likedGenres.size * candidateGenres.size);

          // Cosine similarity
          const score = denom > 0 ? intersectionSize / denom : 0;

          return { ...candidate, score };
        });

        // Step 5: Sort by score
        scoredMovies.sort((a, b) => b.score - a.score);

        // Step 6: Top 2
        const topRecommendations = scoredMovies.slice(0, 2);

        // Step 7: Display
        if (topRecommendations.length > 0) {
          renderRecommendations(likedMovie, topRecommendations);
          const titles = topRecommendations.map(m => m.title).join(', ');
          resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend (Cosine): ${titles}`;
          resultElement.className = 'success';
        } else {
          renderRecommendations(likedMovie, []);
          resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
          resultElement.className = 'error';
        }
      } catch (error) {
        console.error('Error in recommendation calculation:', error);
        resultElement.textContent = "An error occurred while calculating recommendations.";
        resultElement.className = 'error';
        renderRecommendations(null, []);
      }
    }, 60);
  } catch (error) {
    console.error('Error in getRecommendations:', error);
    resultElement.textContent = "An unexpected error occurred.";
    resultElement.className = 'error';
    renderRecommendations(null, []);
  }
}
