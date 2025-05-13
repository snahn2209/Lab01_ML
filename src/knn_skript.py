import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from cifar_loader import load_training_data, load_test_data, show_image


training_data_path = r"../cifar-10-batches-py"
test_data_path = r"../cifar-10-batches-py"

"""
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict
"""


class KNN:
    def __init__(self, k=3, distance='l2'):
        self.k = k   # k is the number of nearest neighbours we take
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, x):
        if self.distance == 'l1':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:  # default: L2
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))


    def _predict(self, x):
        distances = self._distance(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def predict(self, X):
        return np.array([self._predict(x) for x in X])


print("Lade Trainingsdaten...")
train_data, train_labels = load_training_data(training_data_path)

print("Lade Testdaten...")
test_data, test_labels = load_test_data(test_data_path)

# Nur ein Teil der Testdaten verwenden, um Rechenzeit zu sparen
test_sample = test_data[:100]
test_sample_labels = test_labels[:100]

# 2. Teste verschiedene k-Werte und Distanzmaße
ks = [1, 3, 5, 7]
distance_types = ['l1', 'l2']
results = {}

for dist in distance_types:
    print(f"\nDistanzmaß: {dist}")
    accs = []
    for k in ks:
        print(f" - K={k} wird berechnet...")
        knn = KNN(k=k, distance=dist)
        knn.fit(train_data, train_labels)
        preds = knn.predict(test_sample)
        acc = accuracy_score(test_sample_labels, preds)
        print(f"   -> Accuracy: {acc:.2f}")
        accs.append(acc)
    results[dist] = accs

# 3. Visualisiere die Accuracy
plt.figure(figsize=(8, 5))
for dist in distance_types:
    plt.plot(ks, results[dist], marker='o', label=f'Distanz: {dist.upper()}')
plt.title("K vs. Accuracy für L1 und L2")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(ks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# 4. Zeige erste 10 Testbilder mit Vorhersagen
knn = KNN(k=3, distance='l2')
knn.fit(train_data, train_labels)
preds = knn.predict(test_data[:10])

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes):
    img = show_image(test_data[i])
    ax.imshow(img)
    ax.set_title(f'KNN: {preds[i]}\nGT: {test_labels[i]}')
    ax.axis('off')
plt.suptitle("Erste 10 Testbilder mit Vorhersagen (K=3, L2)")
plt.tight_layout()
plt.show()

"""
# Extract the data and labels
train_data = train_data_dict[b'data']
train_labels = np.array(train_data_dict[b'labels'])
test_data_path = test_data_dict[b'data']
test_labels = np.array(test_data_dict[b'labels'])

# Create and train KNN
knn = KNN(k=5)
knn.fit(train_data, train_labels)

# Predict on all test images
# predictions = knn.predict(test_data)

# Predict on the first 50 test images
predictions = knn.predict(test_data_path[:50])

# Visualize the first 10 test images with their predicted labels
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes):
    img = test_data_path[i].reshape(3, 32, 32).transpose(1, 2, 0)  # reshape and permute to 32x32x3
    ax.imshow(img)
    ax.set_title(f'Label: {predictions[i]}')
    ax.axis('off')
plt.show(block=False)()

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
plt.show(block=False)()
"""