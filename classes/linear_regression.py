import numpy as np
"""
A linear regression class that implements the fitting and predicting process in a class.
It will use a vectorized format so that it can scale to more predictors and paramters.

A formal definition for a linear regression model is y_hat = X*B
where:
- X = the Design matrix that will allow the predictor values to be muliplied by their parameters
- Y_hat = the predictions vector made from the Design Matrix * Beta vector(parameters)
           An (n x 1) column vector with n predictions.
- B = the parameter vector that contains Beta_0,Beta_1 ... Beta_n. These will be optimized
      to reduce the cost function. Gradient descent will be done in a vectorized manner here.
-
"""

class LinearRegression:
    def __init__(self, learning_rate = 0.001, epochs= 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Beta = None

    """
    fit(X,y) finds the parameters that will fit the best line.
    It assumes that even though there is one feature, it should be a np matrix and not
    a 1D vector. The same applies for the Y vector, which should also be a np matrix.

    """
    def fit(self,X,Y):
        # Ensure a list or dataframe will turn to a np array
        X = np.array(X)
        Y = np.array(Y)
        # If the features vector has only one feature, turn it into a Matrix with Rx1 dimensions
        if X.ndim == 1:
            X=X.reshape(-1,1)
        Y = Y.reshape(-1,1)

        n_observations = X.shape[0]
        # Design matrix with a column of 1s, and each other column is the respective feature column
        X_design = np.hstack((np.ones((n_observations,1)),X))
        # A parameter column vector (np matrix) that is initialised to 0 each parameter
        self.Beta = np.zeros((X_design.shape[1],1))
        X_T = X_design.T
        # Now we find the best beta parameters
        for epoch in range(self.epochs):
            Y_hat = X_design @ self.Beta
            residuals = Y - Y_hat
            d_Beta = (-2/n_observations)*(X_T @ (residuals))
            self.Beta = self.Beta - self.learning_rate*d_Beta
        return self
        
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_observations = X.shape[0]
        X_design = np.hstack((np.ones((n_observations, 1)), X))
        return X_design @ self.Beta

