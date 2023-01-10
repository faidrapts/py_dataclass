import numpy as np
from sklearn import neighbors
from joblib import dump, load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gs
import struct
from array import array
import os


# function to load data
def load_mnist(folder='./__files', train=True):

    for file in os.listdir(folder):
        print(file)
        with open(f"{folder}/{file}", "rb") as readfile:

            if train == True:

                if ("images" in file) and ("train" in file):
                    magic, size, rows, cols = struct.unpack(
                        ">IIII", readfile.read(16))
                    if magic != 2051:
                        raise ValueError('Magic number mismatch, expected 2051,'
                                         'got {}'.format(magic))

                    image_data = array("B", readfile.read())
                    images = []
                    for i in range(size):
                        images.append([0] * rows * cols)

                    for i in range(size):
                        images[i][:] = image_data[i *
                                                  rows * cols:(i + 1) * rows * cols]

                if ("labels" in file) and ("train" in file):
                    magic, size = struct.unpack(">II", readfile.read(8))
                    if magic != 2049:
                        raise ValueError('Magic number mismatch, expected 2049,'
                                         'got {}'.format(magic))

                    labels = array("B", readfile.read())
            else:

                if ("images" in file) and not ("train" in file):
                    magic, size, rows, cols = struct.unpack(
                        ">IIII", readfile.read(16))
                    if magic != 2051:
                        raise ValueError('Magic number mismatch, expected 2051,'
                                         'got {}'.format(magic))

                    image_data = array("B", readfile.read())
                    images = []
                    for i in range(size):
                        images.append([0] * rows * cols)

                    for i in range(size):
                        images[i][:] = image_data[i *
                                                  rows * cols:(i + 1) * rows * cols]

                if ("labels" in file) and not ("train" in file):
                    magic, size = struct.unpack(">II", readfile.read(8))
                    if magic != 2049:
                        raise ValueError('Magic number mismatch, expected 2049,'
                                         'got {}'.format(magic))

                    labels = array("B", readfile.read())

    return images, labels


if __name__ == "__main__":

    nn_number_values = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    train_images, train_labels = load_mnist()
    test_images, test_labels = load_mnist(train=False)
    np_train_labels = np.array(train_labels)
    np_test_labels = np.array(test_labels)

    fig_grid = plt.figure()
    grid = gs(3, 3)
    np_train_img = np.array(train_images)
    for i in range(3):
        for j in range(3):
            ax = fig_grid.add_subplot(grid[i, j])
            ax.imshow(
                (np.reshape(np_train_img[i*10+j], (28, 28))*255), cmap=plt.get_cmap('gray'))
    plt.savefig('numbers.pdf')

    accuracy_scores = []
    best_accuracy = 0
    for n in nn_number_values:
        knn = neighbors.KNeighborsClassifier(n_neighbors=n)
        knn.fit(train_images[:1200], np_train_labels[:1200])
        predicted_labels = knn.predict(test_images)

        accuracy_score = knn.score(test_images, np_test_labels)
        accuracy_scores.append(accuracy_score)
        print(accuracy_score)

        if accuracy_score > best_accuracy:
            best_accuracy = accuracy_score
            knn_best = knn

    i = int(np.argmax(accuracy_scores))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" %
          (nn_number_values[i], accuracy_scores[i] * 100))

    dump(knn_best, 'model.sk')

    test_correct_model_saved = load('model.sk')
    print(test_correct_model_saved.score(test_images, np_test_labels))

    fig = plt.figure()
    ax_x1 = fig.add_subplot()
    ax_x1.plot(nn_number_values, accuracy_scores, c='purple')

    ax_x1.set_xlim([0, 21])
    ax_x1.set_ylim([0, 1])
    ax_x1.set_xticks(nn_number_values)
    ax_x1.set_title('kNN Test Accuracy')
    ax_x1.set_xlabel('Number of NN')
    ax_x1.set_ylabel('Test Accuracy')

    plt.savefig('knn.pdf')
