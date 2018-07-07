from .imps import *

class RMSE:
    def __init__(self):
        self.mses = []

    def __call__(self, predict, y):
        self.mses.append(F.mse_loss(predict, y).item())
    
    def res(self, bses):
        return np.sqrt(np.average(self.mses, weights=bses))