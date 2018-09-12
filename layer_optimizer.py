from .imps import *

class LayerOptimizer:
    def __init__(self, model, opt_fn=torch.optim.SGD, **opt_args):
        layer_groups = model.get_layer_groups()
        param_group = [{'params': self.trainable_lg_params(lg)} for lg in layer_groups]
        self.opt = opt_fn(param_group, lr=0, **opt_args)
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
        """lg: layer_group, list of pytorch module
        """
        return chain(*[self.trainable_params(l) for l in lg])
    
    def lg_params(self, lg):
        """lg: layer_group, list of pytorch module
        """
        return chain(*[m.parameters() for m in lg])

    def __len__(self):
        return len(self.layer_groups)
    