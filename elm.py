import numpy as np

class Elm:
    def __init__(self, n, a=None, random=None):
        if random:
            np.random.seed(random)
        self.n = n
        self.a = a
        self.w = None
        self.b = None
        self.out_w = None

    def fit(self, X, Y):
        features = X.shape[1] if len(X.shape) > 1 else 1
        self.w = np.random.rand(self.n, features) * 2 - 1
        self.b = np.random.rand(self.n) * 2 - 1
        self.out_w = self.weights(self.outputs(X), Y)

    def predict(self, X):
        return np.dot(self.outputs(X), self.out_w)

    def weights(self, out, Y):
        if self.a:
            inv = np.linalg.pinv(np.dot(out.T, out) + np.eye(self.n)*self.a)
            t = np.dot(inv, out.T)
        else:
            t = np.linalg.pinv(out)
        return np.dot(t, Y)

    def outputs(self, X):               
        out = np.zeros((X.shape[0], self.n))
        for i, x in enumerate(X):
            for j, w in enumerate(self.w):
                out[i, j] = self.sigmoid(np.dot(w, x) + self.b[j])
        return out

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
