import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay


def find_hyperparams(base_model, paramgrid, features, targets, cv=5, **kwopts) -> GridSearchCV:

    opt_model = GridSearchCV(base_model, paramgrid, cv=cv, n_jobs=3)
    opt_model.fit(features, targets)

    return opt_model


if __name__ == '__main__':

    iris = load_iris(as_frame=True)
    X_all = iris.data
    X = X_all.drop(columns=['petal length (cm)', 'petal width (cm)'])
    y = iris.target
    df = pd.concat([X, y], axis=1)
    labels = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True)

    train_scores = []
    test_scores = []

    knn = KNeighborsClassifier()
    knn_opt = find_hyperparams(
        knn, {'n_neighbors': [5, 10, 20, 25, 30, 35, 40, 50, 60, 75]}, X_train, y_train)
    y_predtrain = knn_opt.predict(X_train)
    y_pred = knn_opt.predict(X_test)
    train_sc = np.round(accuracy_score(y_train, y_predtrain), 2)
    test_sc = np.round(accuracy_score(y_test, y_pred), 2)
    train_scores.append(train_sc)
    test_scores.append(test_sc)

    svc_linear = SVC(kernel='linear')
    svc_linear_opt = find_hyperparams(
        svc_linear, {'C': [0.001, 0.01, 0.1, 1]}, X_train, y_train)
    y_predtrain = svc_linear_opt.predict(X_train)
    y_pred = svc_linear_opt.predict(X_test)
    train_sc = np.round(accuracy_score(y_train, y_predtrain), 2)
    test_sc = np.round(accuracy_score(y_test, y_pred), 2)
    train_scores.append(train_sc)
    test_scores.append(test_sc)

    svc_polynomial = SVC(kernel='poly')
    svc_polynomial_opt = find_hyperparams(
        svc_polynomial, {'C': [0.001, 0.01, 0.1, 1], 'degree': [3, 4, 5], 'coef0': [0, 1]}, X_train, y_train)
    y_predtrain = svc_polynomial_opt.predict(X_train)
    y_pred = svc_polynomial_opt.predict(X_test)
    train_sc = np.round(accuracy_score(y_train, y_predtrain), 2)
    test_sc = np.round(accuracy_score(y_test, y_pred), 2)
    train_scores.append(train_sc)
    test_scores.append(test_sc)

    svc_rbf = SVC(kernel='rbf')
    svc_rbf_opt = find_hyperparams(
        svc_rbf, {'C': [0.001, 0.01, 0.1, 1, 10, 100]}, X_train, y_train)
    train_scores.append(train_sc)
    y_predtrain = svc_rbf_opt.predict(X_train)
    y_pred = svc_rbf_opt.predict(X_test)
    train_sc = np.round(accuracy_score(y_train, y_predtrain), 2)
    test_sc = np.round(accuracy_score(y_test, y_pred), 2)
    test_scores.append(test_sc)

    fig_grid, sub = plt.subplots(2, 2)

    models = [knn_opt, svc_linear_opt, svc_polynomial_opt, svc_rbf_opt]
    title_list = ['knn', 'SVC linear', 'SVC polynomial', 'SVC rbf']

    for model, name, ax, train_score, test_score in zip(models, title_list, sub.flatten(), train_scores, test_scores):
        plot = DecisionBoundaryDisplay.from_estimator(
            model, X, response_method='predict', grid_resolution=200, ax=ax)

        ax.scatter(df.loc[df['target'] == 0, 'sepal length (cm)'], df.loc[df['target']
                   == 0, 'sepal width (cm)'], s=7, marker='.', c='blue', label=labels[0])
        ax.scatter(df.loc[df['target'] == 1, 'sepal length (cm)'], df.loc[df['target']
                   == 1, 'sepal width (cm)'], s=7, marker="<", c='purple', label=labels[1])
        ax.scatter(df.loc[df['target'] == 2, 'sepal length (cm)'], df.loc[df['target']
                   == 2, 'sepal width (cm)'], s=7, marker='s', c='green', label=labels[2])
        ax.set_title(
            f"{name}, train: {train_score}, test: {test_score} \n {model.best_params_}")
        ax.legend(loc='upper right', fontsize='x-small')

    fig_grid.tight_layout(pad=1)
    plt.savefig('plot.pdf')
    plt.show()
