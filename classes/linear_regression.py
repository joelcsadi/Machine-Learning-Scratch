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
    def __init__(self, learning_rate = 0.001, epochs= 1000, mode = "batch"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Beta = None
        self.mode = mode

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

        # Now we find the best beta  using different optimization modes

        if self.mode == "batch":
            for epoch in range(self.epochs):
                Y_hat = X_design @ self.Beta
                residuals = Y - Y_hat
                d_Beta = (-2/n_observations)*(X_T @ (residuals))
                self.Beta = self.Beta - self.learning_rate*d_Beta
                if not np.isfinite(self.Beta).all():
                    raise ValueError("Divergence in training detected. Try reducing learning rate or increasing epochs.")

        elif self.mode == "normal":
            self.Beta = np.linalg.inv(X_T @ X_design) @ X_T @ Y

        elif self.mode == "stochastic":
            for epoch in range(self.epochs):
                indices_array = np.random.permutation(n_observations)
                for i in indices_array:
                    X_design_i = X_design[i,:].reshape(1,-1)
                    Y_i = Y[i]
                    Y_hat_i =X_design_i @ self.Beta
                    residual_i = Y_i - Y_hat_i
                    d_Beta = -2*(X_design_i.T @ residual_i)
                    self.Beta = self.Beta - self.learning_rate*d_Beta

                    if not np.isfinite(self.Beta).all():
                        raise ValueError("Divergence in training detected. Try reducing learning rate or increasing epochs.")
        else:
            raise ValueError("Invalid mode selected. Choose from 'batch', 'stochastic' or 'normal'")
        return self

        

    """
    predict(X) uses the parameters fitted on the model uses it to predict the reponse
    given the features matrix X.
    """
    def predict(self, X):
        if self.Beta is None:
            raise ValueError("Model is not fitted yet. Please call fit(X,y) before predicting")
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_observations = X.shape[0]
        X_design = np.hstack((np.ones((n_observations, 1)), X))
        return X_design @ self.Beta

    """
    get_R_squared(X,Y) obtains the percentage of variation that the fitted model explains
    by comparing and subtracting it of the total variaton of the model mean(y_bar).
    It calculates the variation around the points and the fitted line(Sum of square Residuals /RSS)
    and calculates the variation around the mean which is(Total sum of squares TSS)

    The formula for R^2 = 1-(RSS/SST)
    It's a metric for goodness of fit.

    """
    def get_R_squared(self,X,Y):
        if self.Beta is None:
            raise ValueError("Model is not fitted yet. Please call fit(X,y) before calculating R squared")
        X = np.array(X)
        Y = np.array(Y)
        # If the features vector has only one feature, turn it into a Matrix with Rx1 dimensions
        if X.ndim == 1:
            X=X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        n_observations = X.shape[0]
        X_design = np.hstack((np.ones((n_observations,1)),X))
        y_hat = self.predict(X)
        y_bar = np.mean(Y)
        RSS = np.sum((Y-y_hat)**2)
        SST = np.sum((Y-y_bar)**2)
        r_squared = 1 - (RSS/SST)

        return r_squared
    
    """
    get_Parameters() returns the parameter vector that were calculated within the fit of the
    model. It goes from Beta_0 ... Beta_n where n is the nth parameter. It includes the
    y_intercept/bias(Beta_0) and the features/weights(Beta_1 -> Beta_n) of the models.

    They were calculated using batch gradient descent in fit(X,Y)
    """
    def get_Parameters(self):
        if self.Beta is None:
            raise ValueError("Model is not fitted yet. Please call fit(X,y) before retrieving parameters")
        return self.Beta