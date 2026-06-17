import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        def linear(w, b, x):
            return x@w + b
        
        for i in range(len(weights)):
            x = linear(weights[i],biases[i], x)
            if i < len(weights) -1 :
                x = np.maximum(x,0)
        
        return x
