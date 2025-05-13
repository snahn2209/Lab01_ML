import numpy as np
import matplotlib.pyplot as plt # for creating visualizations
import time
from sklearn.metrics import accuracy_score
from cifar_loader import load_training_data, load_test_data, show_image

# define class names to return names instead of label numbers
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# load data and create test sample
print("load trainings data...")
train_data, train_labels = load_training_data("../cifar-10-batches-py")
print("load test data...")
test_data, test_labels = load_test_data("../cifar-10-batches-py")
sample_size = 100
test_sample = test_data[:sample_size,]
test_sample_labels = test_labels[:sample_size,]

class KNN:
    def __init__(self, k=3, distance='l2'):
        self.k = k   # k is the number of nearest neighbours we take
        self.distance = distance

    # saves trainingdata X and associated labels y
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, x):
        if self.distance == 'l1':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:  # default: L2
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    # private method for single prediction
    def _predict(self, x):
        distances = self._distance(x)
        #get k-nearest neighbours
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        #return most common class
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    #public method for predicting multiple samlpes
    def predict(self, X):
        return np.array([self._predict(x) for x in X])

# Test parameters
ks = [1, 3, 5, 7]
distance_types = ['l1', 'l2']
"""
(for better understanding)
-> empty dictionary (like HashMap in java)
- The dictionary stores accuracy scores for different combinations of:
    - Distance metrics (L1 and L2)
    - K values (1, 3, 5, 7)

- Each distance type (L1 or L2) is a key in the dictionary
- The corresponding value is a list of accuracy scores for different k values
"""
results = {}

start_time = time.time()
# for each distance metric (l1/l2)
for dist in distance_types:
    # create figure
    fig, axes = plt.subplots(len(ks), 10, figsize=(20, 8))
    fig.suptitle(f"predictions with {dist.upper()} for k=1,3,5,7")
    
    for k_idx, k in enumerate(ks):
        print(f"computing {dist} for k={k}")
        #create and train knn model
        knn = KNN(k=k, distance=dist)
        knn.fit(train_data, train_labels)
        
        # make prredictions for Test sample
        preds = knn.predict(test_sample)
        # calculate accuracy
        acc = accuracy_score(test_sample_labels, preds) #returns float between 0.0 and 1.0 (compares test_sample_labels with actual predictions)
        results.setdefault(dist, []).append(acc)
        
        # Plot first 10 predictions
        for i in range(10):
            ax = axes[k_idx, i]
            img = show_image(test_sample[i])
            ax.imshow(img)
            pred_name = class_names[preds[i]]
            true_name = class_names[test_sample_labels[i]]
            ax.set_title(f'Pred: {pred_name}\nTrue: {true_name}')
            ax.axis('off')

    plt.tight_layout()

# accuracy comparison
plt.figure(figsize=(8, 5))
for dist in distance_types:
    plt.plot(ks, results[dist], marker='o', label=f'Distance: {dist.upper()}')
plt.title("K vs. Accuracy for L1 and L2")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(ks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ time needed: {elapsed_time:.2f} sec")