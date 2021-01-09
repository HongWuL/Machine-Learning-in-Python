import numpy as np

class Perceptron:
    def __init__(self, eta = 0.1, max_iter = 1000):
        """

        """
        self.w = None
        self.b = None
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, X, y):
        """
        :param X: training examples with shape (n_samples, n_features)
        :param y: labels of training examples, a vector with length n_samples
        :return:
        """
        self.w = np.zeros(X.shape[1])
        self.b = 0

        iter = 0
        while iter < self.max_iter:
            t = -1
            for i in range(X.shape[0]):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    t = i
                    break
            if t == -1:
                return
            self.w += self.eta * y[t] * X[t]
            self.b += self.eta * y[t]
            iter += 1


    def predict(self, X):
        """
        :param X: testing examples with shape (n_test, n_features)
        :return:
        """
        y_p = []
        for x in X:
            r = 1 if np.dot(self.w, x) + self.b > 0 else -1
            y_p.append(r)
        return np.array(y_p), self.w, self.b

class PerceptionDual:
    def __init__(self, max_iter = 1000):
        self.a = None
        self.b = None
        self.w = None
        self.gram = None
        self.max_iter = max_iter

    def fit(self, X, y):

        n_samples = X.shape[0]
        self.a = np.zeros((n_samples))
        self.b = 0
        self.gram = np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(i + 1):
                self.gram[i][j] = np.dot(X[i], X[j])
                self.gram[j][i] = self.gram[i][j]

        iter = 0
        while iter < self.max_iter:
            t = -1
            for i in range(n_samples):
                l = 0
                for j in range(n_samples):
                    l += self.a[j] * y[j] * self.gram[j][i]
                if y[i] * l <= 0:
                    t = i
            if t == -1:
                self.w = np.zeros(X.shape[1])
                for k in range(n_samples):
                    self.w += self.a[k] * y[k] * X[k]
                return
            self.a[t] += 1
            self.b += y[t]
            iter += 1

    def predict(self, X):
        """
        :param X: testing examples with shape (n_test, n_features)
        :return:
        """
        y_p = []

        for x in X:
            r = 1 if np.dot(self.w, x) + self.b > 0 else -1
            y_p.append(r)
        return np.array(y_p), self.w, self.b




