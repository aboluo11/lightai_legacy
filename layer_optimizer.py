from .imps import *

class LayerOptimizer:
    def __init__(self, layer_groups, opt_fn):
        param_group = [{'params': self.trainable_lg_params(lg)} for lg in layer_groups]
        self.opt = opt_fn(param_group)
        self.layer_groups = layer_groups
    
    def set_lrs(self, lrs):
        for lr,pg in zip(lrs,self.opt.param_groups):
            pg['lr'] = lr
        self.lrs = lrs

    def set_wds(self, wds):
        for wd,pg in zip(wds,self.opt.param_groups):
            pg['weight_decay'] = wd
        self.wds = wds
    
    def trainable_params(self, m):
        return [p for p in m.parameters() if p.requires_grad]
    
    def trainable_lg_params(self, lg):
        if isinstance(lg, (list, tuple)):
            return chain(*[self.trainable_params(l) for l in lg])
        return self.trainable_params(lg)
    