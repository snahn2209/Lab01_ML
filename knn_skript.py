import numpy as np
import pickle
import matplotlib.pyplot as plt


training_data = r"/Users/karolinaserova/Lab01_ML/cifar-10-batches-py/data_batch_1"
test_data = r"/Users/karolinaserova/Lab01_ML/cifar-10-batches-py/test_batch"

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

class KNN:
    def __init__(self, k=3):
        self.k = k
        # k is the number of nearest neighbours we take

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1)) # euclidian distance formula
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Load data directly using unpickle
train_data_dict = unpickle(training_data)
test_data_dict = unpickle(test_data)

# Extract the data and labels
train_data = train_data_dict[b'data']
train_labels = np.array(train_data_dict[b'labels'])
test_data = test_data_dict[b'data']
test_labels = np.array(test_data_dict[b'labels'])

# Create and train KNN
knn = KNN(k=5)
knn.fit(train_data, train_labels)

# Predict on all test images
# predictions = knn.predict(test_data)

# Predict on the first 50 test images
predictions = knn.predict(test_data[:50])

# Visualize the first 10 test images with their predicted labels
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes):
    img = test_data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # reshape and permute to 32x32x3
    ax.imshow(img)
    ax.set_title(f'Label: {predictions[i]}')
    ax.axis('off')
plt.show()
















# Count the occurrences of each label in predictions (for the first 50)
unique, counts = np.unique(predictions, return_counts=True)
pred_counts = dict(zip(unique, counts))

# Count the occurrences of each label in actual labels (for the first 50)
unique, counts = np.unique(test_labels[:50], return_counts=True)
actual_counts = dict(zip(unique, counts))

# Labels and counts for predicted and actual
labels = range(10)  # CIFAR-10 has 10 classes
pred_freq = [pred_counts.get(label, 0) for label in labels]
actual_freq = [actual_counts.get(label, 0) for label in labels]

# Create a scatter plot to show distribution of Predicted vs. Actual labels
plt.figure(figsize=(10, 6))
plt.scatter(labels, pred_freq, color='blue', alpha=0.5, label='Predicted')
plt.scatter(labels, actual_freq, color='red', alpha=0.5, label='Actual')
plt.title('Distribution of Predicted vs. Actual Labels')
plt.xlabel('Class Labels')
plt.ylabel('Frequency')
plt.xticks(labels)  # Ensure labels match the class indices
plt.legend()
plt.grid(True)
plt.show()