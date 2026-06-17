import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)

        def sigmoid(x):
            return 1/(1+np.exp(-x))
        def gradient_sg(x):
            return sigmoid(x)*(1-sigmoid(x))
        
        pre_act = np.dot(w,x) + b 
        y_hat = 1/(1+np.exp(-pre_act))

        L = 0.5*((y_hat-y_true)**2)

        dL_dw = (y_hat-y_true)*gradient_sg(pre_act)*x
        dL_db = (y_hat-y_true)*gradient_sg(pre_act)

        return (np.round(dL_dw, 5), np.round(dL_db, 5))


