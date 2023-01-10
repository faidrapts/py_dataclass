import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


def breit_wigner(x, a, b, c):
    x = np.squeeze(x)
    g = a / ((b-x)**2 + c)
    return g.tolist()


def eval_jacobian(x, func, params, h=0.0001):
    J = np.zeros((x.shape[0], len(params)))
    if len(params) == 3:
        a, b, c = params
        J[:, 0] = (np.array(func(x, a+h, b, c)) -
                   np.array(func(x, a-h, b, c)))/(2*h)
        J[:, 1] = (np.array(func(x, a, b+h, c)) -
                   np.array(func(x, a, b-h, c)))/(2*h)
        J[:, 2] = (np.array(func(x, a, b, c+h)) -
                   np.array(func(x, a, b, c-h)))/(2*h)

    else:
        array = np.array(params)

        for i, p in enumerate(params):
            array = np.array(params)
            array[i] = float(p) + h
            first = array.tolist()
            array[i] = float(p) - h
            second = array.tolist()
            J[:, i] = (np.array(func(x, *first)) -
                       np.array(func(x, *second)))/(2*h)

    return J


def eval_errors(x_all: np.array, y_all, func, params: list):
    return y_all - func(x_all, *list(params))


def get_params(jacobian, lmalambda, error):
    square = jacobian.T@jacobian
    deltaparams = np.linalg.inv(
        square + lmalambda*np.identity(square.shape[0]))@jacobian.T@error
    return deltaparams


def lma_quality_measure(x, y, func, params, delta_params, jac, lma_lambda):
    new_params = [p+d for p, d in zip(params, delta_params)]
    e = eval_errors(x, y, func, params)
    new_e = eval_errors(x, y, func, new_params)
    rho = (e.T@e-new_e.T@new_e)/(delta_params.T @
                                 (lma_lambda*delta_params+jac.T@e))
    print("real rho:", rho)
    return rho


def lma(X_all, y_all, func, param_guess, **kwargs):
    lma_lambda = None
    while True:
        jac = eval_jacobian(X_all, func, param_guess)

        # calculate lambda if not set
        if lma_lambda is None:
            lma_lambda = np.linalg.norm(jac.T@jac)
        e = eval_errors(X_all, y_all, func, param_guess)
        delta_params = get_params(jac, lma_lambda, e)

        # calculate quality measure
        lma_rho = lma_quality_measure(
            X_all, y_all, func, param_guess, delta_params, jac, lma_lambda)
        if lma_rho > 0.75:
            lma_lambda /= 3
        elif lma_rho < 0.25:
            lma_lambda *= 2
        else:
            lma_lambda = lma_lambda

        # only change parameters if the quality measure is greater 0
        if lma_rho > 0:
            param_guess = [x+d for x, d in zip(param_guess, delta_params)]

        param_change = np.linalg.norm(delta_params)/np.linalg.norm(param_guess)
        if param_change < 0.00001:
            break

    return param_guess


def lin2d(X, y, func, a, b):

    fit1 = lma(X[:, 0], y, func, a)
    fit2 = lma(X[:, 1], y, func, b)
    gx0 = breit_wigner(X[:,0],*fit1)
    gx1 = breit_wigner(X[:,1],*fit2)

    return gx0,gx1


if __name__ == "__main__":

    xtest = np.array([[0], [1]])
    pars = [0.5, 0.2, 1]
    jacobian = eval_jacobian(xtest, breit_wigner, pars)

    print("test lin2d:", lin2d(np.ones((400,2)),np.random.rand(400),breit_wigner,np.random.rand(3),np.random.rand(3)))

    data = pd.read_csv('./__files/breit_wigner.csv')
    x_all = data[["x"]].to_numpy()
    Y_all = data["g"].to_numpy()
    fit = lma(x_all, Y_all, breit_wigner, np.random.rand(3)*1)
    print("Fit:", fit)
    x_vals = np.linspace(0, 200, 200000)
    plt.plot(x_vals, breit_wigner(x_vals, *fit))
    plt.plot(x_all, Y_all, '.', c='purple')
    plt.title('Breit-Wigner function for params a=%.4f, b=%.4f, c=%.4f' %
              (fit[0], fit[1], fit[2]))
    # plt.show()
    plt.savefig('./plot.pdf')
