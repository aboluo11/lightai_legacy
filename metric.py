from .imps import *

class RMSE:
    def __init__(self):
        self.mses = []

    def __call__(self, predict, y):
        self.mses.append(F.mse_loss(predict, y, reduction=None))
    
    def res(self):
        return torch.sqrt(torch.cat(self.mses).mean()).item()