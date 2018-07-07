from .imps import *

class BatchSampler:
    def __init__(self, dataset, bs):
        self.bs = bs
        self.dataset = dataset
        
    def __iter__(self):
        idxs = np.random.permutation(len(self.dataset))
        idxs = [idxs[i:min(i+self.bs, len(idxs))] for i in range(0, len(idxs), self.bs)]
        return iter(idxs)
            
    def __len__(self):
        return (len(self.dataset) + self.bs - 1) // self.bs

class SortishSampler:
    def __init__(self, dataset, bs):
        self.bs = bs
        self.dataset = dataset

    def __iter__(self):
        idxs = np.random.permutation(len(self.dataset))
        sz = self.bs*50
        ck_idxs = [idxs[i:min(i+sz, len(idxs))] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(ck,
         key=lambda x:len(self.dataset.text[x]), reverse=True) for ck in ck_idxs])
        sz = self.bs
        sort_idxs = [sort_idx[i:min(i+sz, len(idxs))] for i in range(0, len(idxs), sz)]
        max_ck = np.argmax([len(self.dataset.text[ck[0]]) for ck in sort_idxs])
        sort_idxs[0], sort_idxs[max_ck] = sort_idxs[max_ck], sort_idxs[0]
        return iter(sort_idxs)

    def __len__(self):
        return (len(self.dataset) + self.bs - 1) // self.bs