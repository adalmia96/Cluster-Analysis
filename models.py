import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
import sys
import matplotlib.pyplot as plt



class Model(object):
    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses



class Perceptron(Model):

    def __init__(self, *, nfeatures):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))

    def fit(self, *, X, y, lr):

        sign = lambda x: (1, -1)[x < 0]
        y = np.where(y==0, -1, y)

        for i in range(len(y)):
            feat = X[i].toarray()
            y_pred = np.dot(feat, self.W).squeeze()
            if (sign(y_pred) != y[i]):
                self.W = self.W + lr * y[i] * feat.T


        #raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        sign = lambda x: int(x >= 0)
        y_pred = np.concatenate( np.dot(X.toarray(), self.W))
        y_sign = np.array([sign(yi) for yi in y_pred])

        return y_sign
        #raise Exception("You must implement this method!")


class WeightedPerceptron(Perceptron):

    def __init__(self, *, nfeatures, alpha_pos, alpha_neg):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def fit(self, *, X, y, lr):
        sign = lambda x: (1, -1)[x < 0]
        alpha = lambda x:(self.alpha_pos, self.alpha_neg)[x < 0]

        y = np.where(y==0, -1, y)
        for i in range(len(y)):
            feat = X[i].toarray()
            y_pred = np.dot(feat, self.W).squeeze()
            if (sign(y_pred) != y[i]):
                self.W = self.W + lr * y[i] * alpha(y_pred) * feat.T


        #raise Exception("You must implement this method!")



class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        for i in range(len(y)):
            feat = X[i].toarray()
            y_pred = self.score(feat)
            if y_pred != (y[i]):
                feat = feat.squeeze()
                self.W[y_pred] = self.W[y_pred] - lr * feat
                self.W[y[i]] = self.W[y[i]] + lr * feat

    def predict(self, X):
        X = self._fix_test_feats(X)
        y_best = np.array([self.score(xi.toarray()) for xi in X])
        return y_best

    def score(self, x_i):
        y_scores = np.dot(self.W, x_i.T)
        return (np.argmax(y_scores))



class Logistic(Model):

    def __init__(self, *, nfeatures):
        super().__init__(nfeatures=nfeatures)
        self.W = np.zeros((nfeatures, 1))

    def fit(self, *, X, y, lr):

        for i in range(X.shape[0]):
            feat = X[i].toarray()
            reg = (np.dot(feat, self.W)).squeeze()

            neg_g_x = self.sigmoid(-1.0*reg)
            g_x = self.sigmoid(reg)

            multiplier_pos = y[i] * neg_g_x
            multiplier_neg = (1.0-y[i]) * g_x * -1.0



            self.W = self.W + lr * (multiplier_pos * feat.T + multiplier_neg * feat.T)



    def predict(self, X):
        X = self._fix_test_feats(X)
        pred = np.zeros(X.shape[0])
        #val = lambda x: (1, 0)[x < 0.5]
        for i in range(X.shape[0]):
            reg = ((np.dot(X[i].toarray(), self.W)).squeeze())
            y_best = self.sigmoid(reg)
            if y_best >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        return pred

        # TODO: Implement this!
        # raise Exception("You must implement this method!")



    def sigmoid(self, logits):
        if logits < 0:
            return np.exp(logits) / (1 + np.exp(logits))
        else:
            return (1 / (1+ np.exp(-1*logits)))
        # TODO: Implement this!
        #raise Exception("You must implement this method!")
