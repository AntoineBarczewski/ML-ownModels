
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression


class lasso_seuil(LinearRegression):
    """
    Ordinary least squares Linear Regression applied on support computed from
    lasso with threshold

    Parameters
    ----------
    alpha : L1 penalisation, by default at 1
    tau : threshold to test coefficients from lasso support.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    Notes
    -----
    From the implementation point of view, it is based on sckitlearn Lasso and OLS.
    """

    def __init__(self, alpha=1, tau=0.1):
        self.alpha = alpha
        self.tau = tau
        self.intercept_ = 0.

    def fit(self, X, y):
        # initialiser le model
        # Lasso simple pour éviter la cross-val à cette étape
        lasso = Lasso(alpha=self.alpha, fit_intercept=False)

        # fit du model sur les données
        lasso_train = lasso.fit(X, y)

        # récupérer les coefficients du lasso
        thetas = lasso_train.coef_

        # initialiser la liste d'indice
        S = []

        # test sur les coefficients
        for j, theta in enumerate(thetas):
            if abs(theta) > self.tau:
                S += [j]

        # initialiser l'ols
        ols = LinearRegression(fit_intercept=False, normalize=False)

        # selectionner sur le support
        X = X[:, S]

        # récupérer l'ouput
        self.coef_, self._residues, self.rank_, self.singular_ = \
                ols.fit(X, y).coef_, ols.fit(X, y)._residues, ols.fit(X, y).rank_, ols.fit(X, y).singular_
        self.coef_ = self.coef_.T

        # récupérer le support
        self.support = S

        return self

    def predict(self, X):

        X = X[:, self.support]

        return self._decision_function(X)
