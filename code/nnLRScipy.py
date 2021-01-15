import random
import numpy as np
from copy import deepcopy
from scipy.sparse import diags
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class nnLRScipy(BaseEstimator, TransformerMixin):

    def __init__(self, reltol=1e-8, maxit=1000, intersection=False, opt_method=None,
                 positive=False, penalty=None, verbose=True):

        self.maxit = maxit
        self.reltol = reltol
        self.verbose = verbose
        self.penalty = penalty
        self.positive = positive
        self.opt_method = opt_method
        self.intersection = intersection

    @staticmethod
    def safe_log10(x, eps=1e-10):
        result = np.where(x > eps, x, -10)
        np.log10(result, out=result, where=result > 0)
        return result

    @staticmethod
    def sigmoid(w, X):
        a = X.dot(w)
        o = expit(a)
        return o

    @staticmethod
    def d_sigmoid(w, X):
        a = X.dot(w)
        o = expit(a)
        do = o * (1 - o)
        return do

    def cost(self, w, X, y, n_samples):
        o = self.sigmoid(w, X)
        c = -(np.vdot(y, self.safe_log10(o)) + np.vdot(1 - y, self.safe_log10(1 - o))) / float(n_samples)
        return c

    def gradient(self, w, X, y, n_samples):
        o = self.sigmoid(w, X)
        grad = -X.T.dot(y - np.expand_dims(o, axis=1)) / float(n_samples)
        return grad

    def hessian(self, w, X, y, n_samples):
        do = self.d_sigmoid(w, X)
        D = diags(do)
        hs = X.T @ D @ X / n_samples
        return hs

    def grad_hess(self, w, X, y, n_samples):
        # gradient AND hessian of the logistic
        grad = self.gradient(w, X, y, n_samples)
        Hs = self.hessian(w, X, n_samples)

        return grad, Hs.reshape(n_samples)

    def fit(self, X, y=None):

        n_classes = len(np.unique(y))

        n_samples, n_features = X.shape

        w = []
        for c in range(0, n_classes):

            y_c = np.array([1 if label == c else 0 for label in y])
            y_c = np.reshape(y_c, (n_samples, 1))

            const, bounds = (), None
            if self.penalty == 'l2':
                const += ({'type': 'eq', 'fun': lambda w: np.vdot(w, w) - 1},)
            elif self.penalty == 'l1':
                const += ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)

            if self.positive:
                # I have tried both
                const += ({'type': 'ineq', 'fun': lambda x: x},) # or
                # bounds = [(0., None)] * n_features
                w_0 = 0.1 * np.ones(n_features)
            else:
                random.seed(c)
                w_0 = np.random.uniform(-1, 1, n_features)

            options = {'disp': self.verbose, 'maxiter': self.maxit}
            f_min = minimize(fun=self.cost, x0=w_0,
                             args=(X, y_c, n_samples),
                             # method=self.opt_method,
                             jac=self.gradient,
                             # hess=self.hessian,
                             tol=self.reltol,
                             constraints=const,
                             bounds=bounds,
                             options=options)

            # f_min = pytron.minimize(self.cost, self.grad_hess, w_0, args=(X, y, n_samples),
            #                     gtol=1e-10, tol=1e-20, max_iter=100)

            w.append(deepcopy(f_min.x))

        self.coef_ = np.vstack(w).T

        return self

    def predict_proba(self, X):
        check_is_fitted(self, msg='not fitted.')

        sigma = self.sigmoid(self.coef_, X)

        return sigma

    def predict(self, X):
        check_is_fitted(self, msg='not fitted.')

        sigma = self.predict_proba(X)
        y_pred = np.argmax(sigma, axis=1)
        return y_pred
