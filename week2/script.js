let model = null;
let isModelReady = false;

window.onload = async () => {
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const predictBtn = document.getElementById('predict-btn');

  try {
    statusEl.textContent = 'Loading MovieLens data...';
    statusEl.className = 'status-box info';
    await loadData();

    populateUserDropdown();
    populateMovieDropdown();
    statusEl.textContent = 'Preparing training data...';

    await trainModel();

    statusEl.textContent = 'Model training completed successfully!';
    statusEl.className = 'status-box success';
    resultEl.textContent = 'Select a user and a movie, then click "Predict Rating".';
    predictBtn.disabled = false;

    predictBtn.addEventListener('click', predictRating);
  } catch (error) {
    console.error('Failed to initialise recommender', error);
    statusEl.textContent = `Initialisation failed: ${error.message}`;
    statusEl.className = 'status-box error';
    resultEl.textContent = 'Unable to train the model due to the error above.';
  }
};

function populateUserDropdown() {
  const select = document.getElementById('user-select');
  select.innerHTML = '<option value="" disabled selected>Select a user</option>';

  userIds.forEach((id) => {
    const option = document.createElement('option');
    option.value = String(id);
    option.textContent = `User ${id}`;
    select.appendChild(option);
  });
}

function populateMovieDropdown() {
  const select = document.getElementById('movie-select');
  select.innerHTML = '<option value="" disabled selected>Select a movie</option>';

  const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
  sortedMovies.forEach((movie) => {
    const option = document.createElement('option');
    option.value = String(movie.id);
    option.textContent = movie.title;
    select.appendChild(option);
  });
}

/**
 * Build the matrix factorisation model.
 * We represent each user and movie as a learnable latent vector (embedding)
 * and take their dot product to estimate the rating. Separate bias embeddings
 * help the model learn systematic tendencies (e.g. harsh users or popular films).
 */
function createModel(numUsersValue, numMoviesValue, latentDim = 20) {
  const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'user-input' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movie-input' });

  const userEmbedding = tf.layers.embedding({
    inputDim: numUsersValue,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'user-embedding'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: numMoviesValue,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'movie-embedding'
  }).apply(movieInput);

  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  const interaction = tf.layers.dot({ axes: 1, name: 'dot-interaction' }).apply([
    userVector,
    movieVector
  ]);
  // Flatten the rank-2 dot output so it matches the 1-D bias tensors below.
  const interactionFlat = tf.layers.flatten({ name: 'interaction-flat' }).apply(interaction);

  const userBias = tf.layers.embedding({
    inputDim: numUsersValue,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'user-bias'
  }).apply(userInput);

  const movieBias = tf.layers.embedding({
    inputDim: numMoviesValue,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'movie-bias'
  }).apply(movieInput);

  const userBiasFlat = tf.layers.flatten({ name: 'user-bias-flat' }).apply(userBias);
  const movieBiasFlat = tf.layers.flatten({ name: 'movie-bias-flat' }).apply(movieBias);

  const prediction = tf.layers.add({ name: 'predicted-rating' }).apply([
    interactionFlat,
    userBiasFlat,
    movieBiasFlat
  ]);

  return tf.model({
    inputs: [userInput, movieInput],
    outputs: prediction,
  });
}

async function trainModel() {
  const statusEl = document.getElementById('status');
  const predictBtn = document.getElementById('predict-btn');
  predictBtn.disabled = true;

  if (ratings.length === 0) {
    throw new Error('Ratings data is empty.');
  }

  model = createModel(numUsers, numMovies);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  // Convert the sparse MovieLens identifiers into dense tensors once to
  // keep the training loop efficient and avoid extra allocations per epoch.
  const userTensor = tf.tensor2d(
    ratings.map((entry) => entry.userIndex),
    [ratings.length, 1],
    'int32'
  );
  const movieTensor = tf.tensor2d(
    ratings.map((entry) => entry.movieIndex),
    [ratings.length, 1],
    'int32'
  );
  const ratingTensor = tf.tensor1d(
    ratings.map((entry) => entry.rating),
    'float32'
  );

  const epochs = 8;
  const batchSize = 128;

  await model.fit([userTensor, movieTensor], ratingTensor, {
    epochs,
    batchSize,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Using tf.nextFrame() keeps the UI responsive while training in the browser.
        await tf.nextFrame();
        statusEl.textContent = `Training model... Epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(4)}`;
        statusEl.className = 'status-box info';
      }
    }
  });

  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();
  isModelReady = true;
}

async function predictRating() {
  if (!model || !isModelReady) {
    return;
  }

  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const userSelect = document.getElementById('user-select');
  const movieSelect = document.getElementById('movie-select');

  const userId = parseInt(userSelect.value, 10);
  const movieId = parseInt(movieSelect.value, 10);

  if (Number.isNaN(userId) || Number.isNaN(movieId)) {
    resultEl.textContent = 'Please choose both a user and a movie before predicting.';
    return;
  }

  const userIndex = userIndexById[userId];
  const movieIndex = movieIndexById[movieId];

  if (userIndex === undefined || movieIndex === undefined) {
    resultEl.textContent = 'Selected user or movie is missing from the dataset.';
    return;
  }

  // Wrap prediction tensors in tf.tidy so we free GPU/CPU memory immediately.
  const rawRating = tf.tidy(() => {
    const userTensor = tf.tensor2d([[userIndex]], [1, 1], 'int32');
    const movieTensor = tf.tensor2d([[movieIndex]], [1, 1], 'int32');

    const predictionTensor = model.predict([userTensor, movieTensor]);
    return predictionTensor.dataSync()[0];
  });

  const clampedRating = Math.min(5, Math.max(1, rawRating));

  const movieTitle = movies.find((movie) => movie.id === movieId)?.title ?? 'the selected movie';
  resultEl.innerHTML = `Predicted rating for <strong>User ${userId}</strong> on <strong>"${movieTitle}"</strong>: ${clampedRating.toFixed(2)} / 5`;
  statusEl.textContent = 'Model training completed successfully!';
  statusEl.className = 'status-box success';
}
