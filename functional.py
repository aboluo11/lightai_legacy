from .imps import *

def T(x, cuda=True):
    if isinstance(x, (list, tuple)):
        return [T(each) for each in x]
    x = np.ascontiguousarray(x)
    if x.dtype in (np.uint8, np.int8, np.int16, np.int32, np.int64):
        x = torch.from_numpy(x.astype(np.int64))
    elif x.dtype in (np.float32, np.float64):
        x = torch.from_numpy(x.astype(np.float32))
    if cuda:
        x = x.pin_memory().cuda(non_blocking=True)
    return x

def listify(x, y):
    """
    :param x: scalar
    :param y: iterable
    """
    return [x] * len(y)

def ratio_listify(x, ratio):
    """
    x: scalar
    ratio: np.array
    """
    return x * ratio
    
def children(m):
    return list(m.children())

def model_cut(m,right):
    return nn.Sequential(*children(m)[:right])

def split_idx(idx_len, percentage, seed):
    np.random.seed(seed)
    idx = np.random.permutation(idx_len)
    train, valid = idx[:int(idx_len*percentage)], idx[int(idx_len*percentage):]
    return train, valid

