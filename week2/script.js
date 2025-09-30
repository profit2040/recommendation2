// The TensorFlow.js model instance is stored globally so the UI handlers and
// training routine can access the same object without passing it around.
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
 * Randomly select up to `limit` rating entries to keep the demo snappy even on
 * CPUs without WebGL acceleration. The sampling still covers a broad range of
 * users and movies, which is sufficient for showcasing the technique.
 */
function getTrainingSubset(allRatings, limit) {
  if (allRatings.length <= limit) {
    return allRatings;
  }

  const shuffled = [...allRatings];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled.slice(0, limit);
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
    // inputDim is (count + 1) to give TensorFlow.js a safe upper bound even if
    // an index equal to numUsersValue appears. The +1 mirrors how Keras handles
    // embedding lookup tables.
    inputDim: numUsersValue + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'user-embedding'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: numMoviesValue + 1,
    outputDim: latentDim,
    embeddingsInitializer: 'heNormal',
    name: 'movie-embedding'
  }).apply(movieInput);

  // Each embedding has shape [batch, 1, latentDim]. Flattening removes the
  // singleton dimension so we are left with a simple latent vector per entity.
  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  // Taking the dot product of the two latent vectors is the core of matrix
  // factorisation—it estimates how strongly the selected user interacts with
  // the selected movie based on learned preferences.
  const interaction = tf.layers.dot({ axes: 1, name: 'dot-interaction' }).apply([
    userVector,
    movieVector
  ]);

  // Bias embeddings let the network capture global tendencies, such as users
  // who rate generously or movies that are universally loved.
  const userBias = tf.layers.embedding({
    inputDim: numUsersValue + 1,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'user-bias'
  }).apply(userInput);

  const movieBias = tf.layers.embedding({
    inputDim: numMoviesValue + 1,
    outputDim: 1,
    embeddingsInitializer: 'zeros',
    name: 'movie-bias'
  }).apply(movieInput);

  const userBiasFlat = tf.layers.flatten().apply(userBias);
  const movieBiasFlat = tf.layers.flatten().apply(movieBias);

  // The final predicted rating is the dot product plus both bias terms.
  const withUserBias = tf.layers.add().apply([interaction, userBiasFlat]);
  const output = tf.layers.add({ name: 'predicted-rating' }).apply([
    withUserBias,
    movieBiasFlat
  ]);

  return tf.model({
    inputs: [userInput, movieInput],
    outputs: output
  });
}

async function trainModel() {
  const statusEl = document.getElementById('status');
  const predictBtn = document.getElementById('predict-btn');
  const resultEl = document.getElementById('result');
  predictBtn.disabled = true;
  isModelReady = false;

  if (ratings.length === 0) {
    throw new Error('Ratings data is empty.');
  }

  // Recreate the model from scratch so repeated visits to the page never reuse
  // stale weights from an earlier training run.
  model = createModel(numUsers, numMovies);
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  const trainingSampleSize = 3000;
  const trainingData = getTrainingSubset(ratings, trainingSampleSize);

  // Convert the sparse MovieLens identifiers into dense tensors once to keep the
  // training loop efficient and avoid extra allocations per epoch.
  // These tensors pack the dense indices and labels so the optimizer can work with them directly.
  const userTensor = tf.tensor2d(
    trainingData.map((entry) => entry.userIndex),
    [trainingData.length, 1],
    'int32'
  );
  const movieTensor = tf.tensor2d(
    trainingData.map((entry) => entry.movieIndex),
    [trainingData.length, 1],
    'int32'
  );
  const ratingTensor = tf.tensor2d(
    trainingData.map((entry) => entry.rating),
    [trainingData.length, 1],
    'float32'
  );

  // Slightly longer training provides noticeably better predictions while still
  // completing quickly in the browser.
  const epochs = 6;
  const batchSize = 128;

  // Train the embeddings jointly. A small epoch count keeps the demo quick
  // while still converging to reasonable rating estimates.
  statusEl.textContent = 'Training model...';
  statusEl.className = 'status-box info';
  resultEl.textContent = 'Training embeddings, please wait...';

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

  // These tensors are no longer needed once the model has been fitted, so we
  // dispose them manually to avoid unnecessary memory growth in long sessions.
  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();
  isModelReady = true;
}

async function predictRating() {
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  if (!model || !isModelReady) {
    resultEl.textContent = 'Model is still training. Please try again in a moment.';
    return;
  }
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
  // The MovieLens IDs coming from the dropdown are remapped to the dense indices
  // that the embedding layers were trained on before the tensors are created.
  const rawRating = tf.tidy(() => {
    const userTensor = tf.tensor2d([[userIndex]], [1, 1], 'int32');
    const movieTensor = tf.tensor2d([[movieIndex]], [1, 1], 'int32');

    const predictionTensor = model.predict([userTensor, movieTensor]);
    return predictionTensor.dataSync()[0];
  });

  // Clamp to the 1–5 star range so outliers produced early in training do not
  // confuse the user-facing UI.
  const clampedRating = Math.min(5, Math.max(1, rawRating));

  const movieTitle = movies.find((movie) => movie.id === movieId)?.title ?? 'the selected movie';
  // Surface a human-readable explanation so learners can connect the prediction to their selection.
  resultEl.innerHTML = `Predicted rating for <strong>User ${userId}</strong> on <strong>"${movieTitle}"</strong>: ${clampedRating.toFixed(2)} / 5`;
  statusEl.textContent = 'Model training completed successfully!';
  statusEl.className = 'status-box success';
}
