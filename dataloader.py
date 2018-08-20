from .imps import *
from .sampler import *

class DataLoader:
    def __init__(self,sampler, batch_tsfm=None):
        self.sampler = sampler
        self.dataset = sampler.dataset
        self.batch_tsfm = batch_tsfm

    def collate(self, batch):
        elem_type = type(batch[0])
        if elem_type.__module__ == 'numpy':
            batch = np.stack(batch)
        elif isinstance(batch[0], collections.Sequence):
            batch = [self.collate(b) for b in zip(*batch)]
        return batch
    
    def get_batch(self, idxs):
        batch = []
        for idx in idxs:
            sample = self.dataset[idx]
            batch.append(sample)
        batch = self.collate(batch)
        if self.batch_tsfm:
            batch = self.batch_tsfm(batch)
        return batch
    
    def __iter__(self):
        for batch in map(self.get_batch, iter(self.sampler)):
            yield batch
        
    def __len__(self):
        return len(self.sampler)