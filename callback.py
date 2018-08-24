from .imps import *

class CallBack:
    def on_train_begin(self): pass
    def on_batch_begin(self): pass
    def on_batch_end(self,loss): pass
    def on_epoch_end(self,trn_loss,vals): pass
    def on_train_end(self): pass

class Recorder(CallBack):
    def __init__(self, layer_opt):
        self.trn_los, self.trn_los_epoch, self.val_los_epoch, self.lrs = [], [], [], []
        self.layer_opt = layer_opt

    def on_batch_end(self, loss):
        self.lrs.append(self.layer_opt.lrs[-1])
        self.trn_los.append(loss)

    def on_epoch_end(self, trn_loss, vals):
        self.trn_los_epoch.append(trn_loss)
        if vals: self.val_los_epoch.append(vals[0])

    def plot_lr(self):
        fig,ax = plt.subplots()
        ax.set_ylabel("lr")
        ax.set_xlabel("iteration")
        ax.plot(self.lrs)
    
    def plot_loss(self):
        fig,ax = plt.subplots()
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        x_axis = range(1,1+len(self.trn_los_epoch))
        ax.plot(x_axis,self.trn_los_epoch,label='train_loss')
        ax.plot(x_axis,self.val_los_epoch,label='val_loss')
        ax.legend()

    def plot_lr_loss(self):
        fig,ax = plt.subplots()
        ax.set_ylabel("loss")
        ax.set_xlabel("learning rate (log scale)")
        ax.set_xscale('log')
        right = -1
        ax.plot(self.lrs[:right], self.trn_los[:right])

class Scheduler(CallBack):
    def __init__(self, layer_opt, wds):
        self.iteration = -1
        self.layer_opt = layer_opt
        if wds is not None:
            wds = np.array(listify(wds, layer_opt.layer_groups))
            self.layer_opt.set_wds(wds)

    def on_batch_begin(self):
        self.iteration += 1

class LR_Finder(Scheduler):
    def __init__(self, start_lrs, end_lrs, wds, nb, layer_opt):
        self.start_lrs = np.array(listify(start_lrs, layer_opt.layer_groups))
        end_lrs = np.array(listify(end_lrs, layer_opt.layer_groups))
        self.best = 1e9
        self.nb = nb
        self.lrs = np.geomspace(start_lrs[-1],end_lrs[-1],num=nb,endpoint=True)
        super().__init__(layer_opt, wds)

    def on_batch_begin(self):
        super().on_batch_begin()
        self.layer_opt.set_lrs(ratio_listify(self.lrs[self.iteration], self.start_lrs))

    def on_batch_end(self, loss):
        if loss < self.best:
            self.best = loss
        if loss > 2*self.best:
            return True
        return False

class CircularLR(Scheduler):
    def __init__(self, layer_opt, peak_lrs, wds, v_ratio, h_ratio, nb, tl_v_pct, tl_h_pct):
        self.peak_lrs = np.array(listify(peak_lrs, layer_opt.layer_groups))
        bottom = self.peak_lrs[-1]/v_ratio
        l = int(nb*(1-tl_h_pct))
        ll = int(l/h_ratio)
        one = np.linspace(bottom, self.peak_lrs[-1], num=ll, endpoint=False)
        two = np.linspace(self.peak_lrs[-1], bottom, num=l-ll, endpoint=False)
        three = np.linspace(bottom, bottom*tl_v_pct, num=nb-l, endpoint=True)
        self.lrs = np.concatenate([one,two,three])
        super().__init__(layer_opt, wds)

    def on_batch_begin(self):
        super().on_batch_begin()
        self.layer_opt.set_lrs(ratio_listify(self.lrs[self.iteration], self.peak_lrs))

class ConstantLR(Scheduler):
    def __init__(self, lrs, wds, layer_opt):
        self.lrs = np.array(listify(lrs, layer_opt.layer_groups))
        super().__init__(layer_opt, wds)

    def on_train_begin(self):
        self.layer_opt.set_lrs(self.lrs)
    
class SaveBestModel(CallBack):
    def __init__(self, learner, small_better, path='model'):
        self.path = Path(f'{path}/best')
        self.best_metric = None
        self.learner = learner
        self.small_better = small_better

    def on_epoch_end(self,trn_loss,vals):
        if not vals: return
        metric = vals[-1]
        if self.small_better:
            metric = -metric
        if not self.best_metric or metric > self.best_metric:
            self.best_metric = metric
            self.learner.save(self.path)

    def on_train_end(self):
        best = -self.best_metric if self.small_better else self.best_metric
        print(f'best metric: {best:.6f}')
