import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        def relu(x):
            return np.maximum(x,0)

        def gradient_r(x):
            mask = x > 0
            grad = np.zeros((len(x)))

            grad[mask] = 1
            return grad
        
        def Linear(x, w, b):
            return np.matmul(w,x) + b

        x = np.array(x)
        w1 = np.array(W1)
        w2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)

        pa1 = Linear(x,W1, b1)
        a1 = relu(pa1)
        y_hat = Linear(a1,W2, b2)
        loss = np.mean((y_hat-y_true)**2)


        delta_2 = (2*(y_hat-y_true))/len(y_hat)
        print(delta_2.shape)
        print(pa1.shape)
        dW2 = np.outer(delta_2, a1)
        db2 = delta_2

        delta_1 = w2.T@delta_2*gradient_r(pa1)
        dW1 = np.outer(delta_1,x)
        db1 = delta_1

        return {'loss': np.round(loss,4), 'dW1': np.round(dW1,4), 'dW2': np.round(dW2,4), 'db1': np.round(db1,4), 'db2': np.round(db2,4)}



        

