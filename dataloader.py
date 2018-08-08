from .imps import *
from .sampler import *
from .text import *

class DataLoader:
    def __init__(self,sampler):
        self.sampler = sampler
        self.dataset = sampler.dataset
    
    def get_batch(self, idxs):
        res = self.dataset[idxs]
        return res
    
    def __iter__(self):
        for batch in map(self.get_batch, iter(self.sampler)):
            yield [T(each) for each in batch]
        
    def __len__(self):
        return len(self.sampler)