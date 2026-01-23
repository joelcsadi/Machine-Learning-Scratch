"""
This program implements a simple linear regression algorithm
in the form:

y_hat = Beta_0 + Beta_1 * x 

    y_hat ~ the predicted value given a predictor x
    Beta_0 ~ the intercept value of the model ie. when x = 0
    Beta_1 ~ the slope value of the model ie. how predictions
            change linearly as x increases per unit
    x ~ independent predictor used to predict the y_hat.

We shall think of y_hat and x in terms of ith observation

We don't know Beta_0 and Beta_1. We need to find the
values that predicts a y.

How do we do that? It has to have the lowest cost function.
Mean squared error is a way to compare different model parameters.
MSE = (Sum(yi-y_hat)^2) / n

where n is the number of observations and yi is ith actual response variable


We can find the parameters that lead to the global minimum cost function.
"""
x = [1,4,5,7,9,14,20]
y = [1,5,12,16,20,24,40]

Beta_0 = 11
Beta_1 = 0.34

y_hat = []
MSE = 0

for i in range(len(x)):
    y_hat.append(Beta_0 + Beta_1*x[i])

print(y_hat)

for i in range(len(y_hat)):
    MSE += (y[i] - y_hat[i])**2 
else:
    MSE = MSE / len(y_hat)
print(MSE)


    





