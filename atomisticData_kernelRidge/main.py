import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('./__files/nitride_compounds.csv', delimiter=',')
y = df['HSE Eg (eV)']
X = df.drop(columns=['Number', 'PBE Eg (eV)', 'HSE Eg (eV)',
                     'Band offset (eV)', 'Formation energy per atom (eV)', 'Nitride'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, test_size=0.2)
krr = KernelRidge()
krr.fit(X_train, y_train)
y_pred = krr.predict(X_test)
print("Score: ", r2_score(y_test, y_pred))

hyperparameter_opt = GridSearchCV(krr, param_grid=[{'alpha': [0.001, 0.01, 0.5, 0.1, 1], 'gamma':[0.001, 0.01, 0.5, 0.1,
                                                                                                  1, 10], 'kernel': ['linear', 'rbf', 'polynomial', 'laplacian', 'chi2', 'sigmoid']}],
                                  cv=10, scoring='neg_mean_squared_error')
hyperparameter_opt.fit(X_train, y_train)
y_pred_opt = hyperparameter_opt.predict(X_test)
y_pred_huh = hyperparameter_opt.predict(X_train)

r2_opt = r2_score(y_test, y_pred_opt)
mae_opt = mean_absolute_error(y_test, y_pred_opt)
print("Score after hyperparameter optimization: ", r2_opt)
print(hyperparameter_opt.best_params_)
dump(hyperparameter_opt, 'model.joblib')

fraction_train_set = np.arange(0.1, 1, 0.1)
mses = []
r2_scores = []

for p in fraction_train_set:
    X_train_fraction, _, y_train_fraction, _ = train_test_split(
        X_train, y_train, test_size=1-p)
    hyperparameter_opt.fit(X_train_fraction, y_train_fraction)
    y_pred_fr = hyperparameter_opt.predict(X_test)
    mses.append(mean_squared_error(y_test, y_pred_fr))
    r2_scores.append(r2_score(y_test, y_pred_fr))


fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax3 = ax1.twinx()
ax1.set_xlabel('Fraction of train set')
ax1.set_title('Learning curves')
ax1.set_ylabel('MSE')
ax3.set_ylabel('${R}^{2}$ Score')
ax1.plot(fraction_train_set, mses, c='purple')
ax3.plot(fraction_train_set, r2_scores, c='blue')
ax3.spines['right'].set_color('blue')
ax3.yaxis.label.set_color('blue')
ax3.tick_params(axis='y', colors='blue')

ax2.set_title('Model R2: %.3f, MAE: %.3f' % (r2_opt, mae_opt))
ax2.set_xlabel('Calculated gap')
ax2.set_ylabel('Predicted gap')

# For the full training set also visualize model performance
# by plotting the predicted HSE-band gap vs. actual calculated band gap

x_values = [0, 1, 2, 3, 4, 5, 6]
y_values = [0, 1, 2, 3, 4, 5, 6]

ax2.plot(x_values, y_values, 'b-')
ax2.scatter(y_train, y_pred_huh, c='blue', s=3, label='training data')
ax2.scatter(y_test, y_pred_opt, c='orange', s=3, label='test data')
ax2.legend(loc='upper left')

# plt.show()
fig.tight_layout(pad=1)
plt.savefig('plot.pdf')
