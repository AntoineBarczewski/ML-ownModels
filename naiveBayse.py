import numpy as np


class NB(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass


    def fit(self, X, y):
        N = X.shape[0]
        class_, class_counts = np.unique(y, return_counts=True)
        prior = class_counts/N

        condprob = []
        for c in class_:
            condprob += [(np.sum(X[y == c], axis=0)+1)/np.sum(np.sum(X[y == c], axis=0)+1)]

        self.class_ = class_
        self.prior = prior
        self.condprob = condprob

        return self


    def predict(self, X):
        condprob = np.log(self.condprob).T

        score_prior = np.tile(np.log(self.prior), X.shape[0]).reshape(-1, len(self.class_))
        score_condprob = X.dot(condprob)

        score = score_prior + score_condprob

        return np.argmax(score, axis=1)


    def score(self, X, y):
        return np.mean(self.predict(X) == y)
