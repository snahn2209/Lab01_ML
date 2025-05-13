import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Funktion zum Laden eines Batches
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        data = data_dict[b'data']
        labels = data_dict[b'labels']
        return data, labels

# a) Lade alle Trainingsdaten
def load_training_data(directory):
    Xtr_list = [] # trainingsdaten
    Ytr_list = [] # labels
    for i in range(1, 6):
        batch_path = os.path.join(directory, f'data_batch_{i}')
        data, labels = load_batch(batch_path)
        Xtr_list.append(data)
        Ytr_list.append(labels)
    Xtr = np.concatenate(Xtr_list)
    Ytr = np.concatenate(Ytr_list)
    return Xtr, Ytr

# b) Lade Testdaten
def load_test_data(directory):
    test_path = os.path.join(directory, 'test_batch')
    data, labels = load_batch(test_path)
    Xte = data
    Yte = np.array(labels)
    return Xte, Yte

# c) Bild aus 3072-Vektor rekonstruieren und anzeigen
def show_image(img_vector):
    img = img_vector.reshape(3, 32, 32).transpose(1, 2, 0)
    return img

"""
# Vorbereitung Test
if __name__ == '__main__':
    cifar_dir = '../cifar-10-batches-py'  # Passe ggf. den Pfad an

    print("Lade Trainingsdaten...")
    Xtr = load_training_data(cifar_dir)
    print("Trainingsdaten geladen:", Xtr.shape)

    print("Lade Testdaten...")
    Y = load_test_data(cifar_dir)
    print("Testlabels geladen:", Y.shape)

    print("Zeige ein Beispielbild...")
    show_image(Xtr[0])  # zeige das erste Bild im Trainingsset
"""