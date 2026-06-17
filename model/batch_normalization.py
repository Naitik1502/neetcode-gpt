import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        y = []
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        momentum = np.array(momentum)
        x_norm = None
        if training :
            bx = np.array(x)
            b_mean = np.mean(x, axis = 0)
            b_var = bx.var(axis = 0)
            x_norm = (x - b_mean)/(np.sqrt(b_var+eps))
            running_mean = ((1-momentum)*running_mean) + (momentum*b_mean)
            running_var = ((1-momentum)*running_var) + (momentum*b_var)
        else :
            x_norm = (x - running_mean)/(np.sqrt(running_var + eps))
        
        y = (gamma*x_norm) + beta
        
        return (np.round(y.tolist(),4), np.round(running_mean.tolist(),4), np.round(running_var.tolist(),4))
            



