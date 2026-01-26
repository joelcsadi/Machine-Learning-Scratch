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
    def __init__(self, learning_rate = 0.01, epochs= 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Beta = None
    
    def fit(self,X,Y):
        n_observations = X.shape[0]
        # Create a design matrix with 1's on the left for intercept and the x values as other cols
        """
        [[1,3,5]    The vectors are column based and combined together in a design matrix
         [1,6,8]
         [1,9,12]]
        """
        X = np.hstack((np.ones((n_observations,1)),X))
        # Create a parameter vector full of zeroes with the number of features as the rows.
        """
        [0 - Beta_0
         0 - Beta_1
         0] - Beta_2 and so on. 

         It will have the same amount of rows as X Design matrix in order to perform matrix
         multiplication
        
        """
        self.Beta = np.zeros((X.shape[1],1))
        Y = Y.reshape(n_observations,1)
        
        # Now we find the best beta parameters
        for epoch in range(self.epochs):
            Y_hat = X @ self.Beta
            residuals = Y - Y_hat
            d_Beta = (-2/n_observations)*(X.T @ (residuals))
            self.Beta = self.Beta - self.learning_rate*d_Beta
        
    def predict(self, X):
        n_observations = X.shape[0]
        X = np.hstack((np.ones(n_observations,1),X))
        return X @ self.Beta

            

