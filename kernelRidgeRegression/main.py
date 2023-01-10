from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import svm
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import json


df = pd.read_csv('./__files/wave.csv')

x_train, x_test, y_train, y_test = train_test_split(
    df['x'], df['y'], shuffle=False, train_size=0.8)

train = pd.concat([x_train, y_train], axis=1)
train.to_csv('train.csv', index=False)

test = pd.concat([x_test, y_test], axis=1)
test.to_csv('test.csv', index=False)


param_grid = {'gamma': [20, 15, 10, 5, 1, 0.1, 0.01, 0.001, 0.0001],
              'alpha': [10, 1, 0.1, 0.01, 0.001, 0.0001]}

X_train = np.array(x_train).reshape(-1, 1)
X_test = np.array(x_test).reshape(-1, 1)

krr = KernelRidge(kernel='rbf')
grid = GridSearchCV(krr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
print(grid.best_params_)

grid_predictions = grid.predict(X_test)
dump(grid, 'model.joblib')

test_mse = mean_squared_error(y_test, grid_predictions)
print(test_mse)
test_mae = mean_absolute_error(y_test, grid_predictions)
test_r2 = r2_score(y_test, grid_predictions)
y_pred_train = grid.predict(X_train)
train_mae = mean_absolute_error(y_train,y_pred_train)
train_mse = mean_squared_error(y_train,y_pred_train)
train_r2 = r2_score(y_train,y_pred_train)

dictionary = {}
dictionary["test_mae"] = test_mae
dictionary["test_mse"] = test_mse
dictionary["test_r2"] = test_r2
dictionary["train_mae"] = train_mae
dictionary["train_mse"] = train_mse
dictionary["train_r2"] = train_r2


with open("scores.json", "w") as outfile:
    json.dump(dictionary, outfile)

x_values_f = np.arange(-10, 10, 0.1)
X_pred_f = x_values_f.reshape(-1,1)
y_values_f = []
for value in x_values_f:
    y = math.exp(-((value/4)**2)) * np.cos(4*value)
    y_values_f.append(y)

fig = plt.figure()
ax_x1 = fig.add_subplot()

ax_x1.plot(x_values_f, y_values_f)
ax_x1.plot(x_values_f, grid.predict(X_pred_f), c='green')
ax_x1.scatter(x_train, y_train, c='blue')
ax_x1.scatter(x_test, y_test, c='pink')

ax_x1.set_xlim([-10.2, 10.2])
ax_x1.set_ylim([-1.1, 1.1])
ax_x1.set_title('MSE: %.3f, MAE: %.3f, ${R}^{2}$: %.3f' % (
    test_mse, test_mae, test_r2))             
ax_x1.legend(["f", "predicted f", "train values",
             "test values"], loc="upper right")

plt.savefig('plot.pdf')
