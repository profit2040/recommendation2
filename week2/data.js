// Global state shared with script.js
let movies = [];
let ratings = [];
let userIds = [];
let numUsers = 0;
let numMovies = 0;
let userIndexById = {};
let movieIndexById = {};

/**
 * Fetch and parse the MovieLens metadata and ratings files.
 * Any UI errors are thrown so the caller can surface them to the user.
 */
async function loadData() {
  // Reset state to avoid leaking values if the function is invoked twice.
  movies = [];
  ratings = [];
  userIds = [];
  numUsers = 0;
  numMovies = 0;
  userIndexById = {};
  movieIndexById = {};

  try {
    const [moviesResponse, ratingsResponse] = await Promise.all([
      fetch('u.item'),
      fetch('u.data')
    ]);

    if (!moviesResponse.ok) {
      throw new Error(`Unable to load u.item (status ${moviesResponse.status})`);
    }

    if (!ratingsResponse.ok) {
      throw new Error(`Unable to load u.data (status ${ratingsResponse.status})`);
    }

    const [moviesText, ratingsText] = await Promise.all([
      moviesResponse.text(),
      ratingsResponse.text()
    ]);

    parseItemData(moviesText);
    parseRatingData(ratingsText);
  } catch (error) {
    console.error('Error while loading MovieLens data', error);
    throw error;
  }
}

/**
 * Parse the MovieLens item metadata (u.item) and populate the movies array.
 * Each movie is stored with its numeric ID and title. We also create an index
 * so the TensorFlow model can work with densely packed IDs.
 */
function parseItemData(text) {
  movies = [];
  movieIndexById = {};

  const lines = text.split('\n');

  for (const line of lines) {
    if (!line.trim()) continue;

    const fields = line.split('|');
    if (fields.length < 2) continue;

    const id = parseInt(fields[0], 10);
    const title = fields[1];

    const index = movies.length;
    movieIndexById[id] = index;
    movies.push({ id, title });
  }

  numMovies = movies.length;
}

/**
 * Parse the MovieLens ratings (u.data) file. The raw IDs are kept for the UI
 * while densely packed indices are generated for the TensorFlow model.
 */
function parseRatingData(text) {
  ratings = [];
  userIndexById = {};

  const userIdSet = new Set();
  const lines = text.split('\n');

  for (const line of lines) {
    if (!line.trim()) continue;

    const fields = line.split('\t');
    if (fields.length < 3) continue;

    const userId = parseInt(fields[0], 10);
    const movieId = parseInt(fields[1], 10);
    const rating = parseFloat(fields[2]);

    if (Number.isNaN(userId) || Number.isNaN(movieId) || Number.isNaN(rating)) {
      continue;
    }

    userIdSet.add(userId);
    ratings.push({ userId, movieId, rating });
  }

  userIds = Array.from(userIdSet).sort((a, b) => a - b);
  userIds.forEach((id, index) => {
    userIndexById[id] = index;
  });
  numUsers = userIds.length;

  // Attach the dense indices so that script.js can train without recalculating them.
  ratings = ratings
    .map((entry) => ({
      ...entry,
      userIndex: userIndexById[entry.userId],
      movieIndex: movieIndexById[entry.movieId]
    }))
    .filter((entry) =>
      entry.userIndex !== undefined && entry.movieIndex !== undefined
    );
}
