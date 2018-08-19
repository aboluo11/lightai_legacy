from .imps import *
from .sampler import *

class DataLoader:
    def __init__(self,sampler, batch_tsfm=None):
        self.sampler = sampler
        self.dataset = sampler.dataset
        self.batch_tsfm = batch_tsfm
    
    def get_batch(self, idxs):
        res = self.dataset[idxs]
        if self.batch_tsfm:
            res = self.batch_tsfm(res)
        return res
    
    def __iter__(self):
        for batch in map(self.get_batch, iter(self.sampler)):
            yield batch
        
    def __len__(self):
        return len(self.sampler)