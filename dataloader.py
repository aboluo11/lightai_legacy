from .imps import *
from .sampler import *
from .text import *

class DataLoader:
    def __init__(self,dataset,text_idx,sampler,bs):
        self.dataset = dataset
        self.batch_sampler = sampler(dataset,bs)
        self.text_idx = text_idx
    
    def get_batch(self, idxs):
        res = self.dataset[idxs]
        if self.text_idx is not None:
            res[self.text_idx] = pad_seqs(res[self.text_idx])
        return res
    
    def __iter__(self):
        for batch in map(self.get_batch, iter(self.batch_sampler)):
            yield [T(sample) for sample in batch]
        
    def __len__(self):
        return len(self.batch_sampler)