import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        n = X.shape[0]
        w = np.zeros((X.shape[1]))
        b = 0

        for i in range(epochs):
            y_hat = np.matmul(X, w) + b
            error = y_hat - y

            L = (1/n)*np.sum(error**2)

            delta_1 = error*(2/n)
            dL_dw = X.T@delta_1
            db_dw = np.sum(delta_1)

            

            w -= lr*dL_dw
            b -= lr*db_dw
        
        return (np.round(w,5), np.round(b,5))


        
