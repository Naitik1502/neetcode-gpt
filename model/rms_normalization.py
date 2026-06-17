import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)

        rmse_term = np.mean(x ** 2) + eps

        rms_x = np.sqrt(rmse_term)
        print(rms_x)

        x_hat = x/rms_x
        print(x)
        print(x_hat)

        return np.round(x_hat*gamma, 4).tolist()
