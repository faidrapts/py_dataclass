import numpy as np
from sklearn.svm import SVC
from joblib import dump
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gs
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import struct
import gzip
from array import array
import os


# function to load data
def load_mnist(folder='./__files', train=True):

    for file in os.listdir(folder):
        print(file)
        with gzip.open(f"{folder}/{file}", "rb") as readfile:

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

    train_images, train_labels = load_mnist()
    test_images, test_labels = load_mnist(train=False)
    np_train_labels = np.array(train_labels)
    np_test_labels = np.array(test_labels)

    svm_rbf = SVC(kernel='rbf')
    c_range = np.logspace(-1,2,4)
    gamma_range = np.logspace(-7,1,9)
    param_grid = dict(C=c_range,gamma=gamma_range)
    rbf_opt = GridSearchCV(svm_rbf,param_grid,n_jobs=-1)
    rbf_opt.fit(train_images[::30],np_train_labels[::30])

    y_pred = rbf_opt.predict(test_images)
    score = accuracy_score(np_test_labels,y_pred)
    print("Accuracy score: %.3f" % score)
    print(rbf_opt.best_params_)
    dump(rbf_opt,'model.joblib')

    fig_grid = plt.figure()
    grid = gs(5, 2)
    np_test_img = np.array(test_images)
    for i in range(5):
        for j in range(2):
            ax = fig_grid.add_subplot(grid[i, j])
            ax.imshow(
                (np.reshape(np_test_img[i*10+j], (28, 28))*255), cmap=plt.get_cmap('gnuplot'))
            ax.set_title(f'Pred: {y_pred[i*10+j]}, True: {np_test_labels[i*10+j]}')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout(pad=0.5)
    plt.savefig('plot.pdf')
