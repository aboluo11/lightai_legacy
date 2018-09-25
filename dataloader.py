from .imps import *
from .sampler import *


class DataLoader:
    def __init__(self, sampler, n_worker=os.cpu_count()//2):
        self.sampler = sampler
        self.dataset = sampler.dataset
        self.n_worker = n_worker

    def collate(self, batch):
        elem_type = type(batch[0])
        if elem_type.__module__ == 'numpy':
            batch = np.stack(batch)
        elif isinstance(batch[0], str):
            return batch
        elif isinstance(batch[0], collections.Sequence):
            batch = [self.collate(b) for b in zip(*batch)]
        return batch

    def get_batch(self, idxs):
        batch = []
        for idx in idxs:
            sample = self.dataset[idx]
            batch.append(sample)
        batch = self.collate(batch)
        return batch

    def __iter__(self):
        if self.n_worker == 0:
            for batch in map(self.get_batch, iter(self.sampler)):
                yield batch
        else:
            with ThreadPoolExecutor(max_workers=self.n_worker) as e:
                for batch in e.map(self.get_batch, iter(self.sampler)):
                    yield batch

    def __len__(self):
        return len(self.sampler)
