import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml


import time

# Generate a synthetic dataset with 100k samples and 1k features
# X, y = make_classification(n_samples=1000000, n_features=1000, n_informative=50, n_redundant=950, n_classes=2, random_state=42)
# X_synthetic, y = make_classification(n_samples=100000, n_features=786, n_informative=50, n_redundant=10, n_classes=10, random_state=42)
# scaler = MinMaxScaler(feature_range=(0, 255))
# X = scaler.fit_transform(X_synthetic)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# X = np.tile(X, (2, 1))  # Repeat X along the row axis
# y = np.tile(y, 2) 
# this would have been the top limit
fraction = 1
# Accept GPU_N and fraction as command-line arguments
if len(sys.argv) != 3:
    print("Usage: python numpy.py <GPU_N> <fraction>")
    sys.exit(1)

GPU_N = int(sys.argv[1])
fraction = float(sys.argv[2])

# Calculate the new number of rows
original_num_rows = X.shape[0]
new_num_rows = int(original_num_rows * fraction)

# Use np.tile to repeat the original array
num_repeats = int(np.ceil(fraction))
tiled_X = np.tile(X, (num_repeats, 1))
tiled_y = np.tile(y, num_repeats)

# Slice to get the desired number of rows
X_resized = tiled_X[:new_num_rows]
y_resized = tiled_y[:new_num_rows]

# Convert data types
# takes circa 2 seconds
X_resized = X_resized.astype(np.float32)
y_resized = y_resized.astype(np.int32)



# print(X.dtype)
# print(y_resized.dtype)


# Update the original variables
X = X_resized
y = y_resized

print(X.nbytes / (1024 ** 2))
print(y.nbytes / (1024 ** 2))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
start_time = time.time()
# Print shapes to verify
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# Train KNN model

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)
end_time = time.time()

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
elapsed_time = end_time - start_time

# Print results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Elapsed time: {elapsed_time:.2f} seconds')


# X_train shape: (800000, 1000)
# X_test shape: (200000, 1000)
# y_train shape: (800000,)
# y_test shape: (200000,)
# Accuracy: 99.29%
# Elapsed time: 116.99 seconds
