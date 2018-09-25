from .imps import *


class CallBack:
    def on_train_begin(self): pass

    def on_batch_begin(self): pass

    def on_batch_end(self, loss, model): pass

    def on_epoch_end(self, trn_loss, vals): pass

    def on_train_end(self): pass


class Recorder(CallBack):
    def __init__(self, layer_opt):
        self.trn_los, self.trn_los_epoch, self.val_los_epoch, self.lrs = [], [], [], []
        self.layer_opt = layer_opt

    def on_batch_end(self, loss, model):
        self.lrs.append(self.layer_opt.lrs[-1])
        self.trn_los.append(loss)

    def on_epoch_end(self, trn_loss, vals):
        self.trn_los_epoch.append(trn_loss)
        if vals: self.val_los_epoch.append(vals[0])

    def plot_lr(self):
        fig, ax = plt.subplots()
        ax.set_ylabel("lr")
        ax.set_xlabel("iteration")
        ax.plot(self.lrs)

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        x_axis = range(1, 1 + len(self.trn_los_epoch))
        ax.plot(x_axis, self.trn_los_epoch, label='train_loss')
        ax.plot(x_axis, self.val_los_epoch, label='val_loss')
        ax.legend()

    def plot_lr_loss(self):
        fig, ax = plt.subplots()
        ax.set_ylabel("loss")
        ax.set_xlabel("learning rate (log scale)")
        ax.set_xscale('log')
        right = -1
        ax.plot(self.lrs[:right], self.trn_los[:right])


class Scheduler(CallBack):
    def __init__(self, layer_opt, wd, wd_ratio):
        self.iteration = -1
        self.layer_opt = layer_opt
        if wd is not None:
            wds = ratio_listify(wd, wd_ratio)
            self.layer_opt.set_wds(wds)

    def on_batch_begin(self):
        self.iteration += 1


class LRFinder(Scheduler):
    def __init__(self, phases, ratio, wd, wd_ratio, layer_opt):
        self.ratio = ratio
        self.best = 1e9
        self.lrs = np.concatenate(phases)
        super().__init__(layer_opt, wd, wd_ratio)

    def on_batch_begin(self):
        super().on_batch_begin()
        self.layer_opt.set_lrs(ratio_listify(self.lrs[self.iteration], self.ratio))

    def on_batch_end(self, loss, model):
        if loss < self.best:
            self.best = loss
        if loss > 2 * self.best:
            return True
        return False


class PhaseLr(Scheduler):
    def __init__(self, phases, ratio, wd, wd_ratio, layer_opt):
        self.lrs = np.concatenate(phases)
        self.ratio = ratio
        super().__init__(layer_opt, wd, wd_ratio)

    def on_batch_begin(self):
        super().on_batch_begin()
        self.layer_opt.set_lrs(ratio_listify(self.lrs[self.iteration], self.ratio))

class Cyclic(Scheduler):
    def __init__(self, phases, ratio, wd, wd_ratio, layer_opt, sv_best_model):
        self.phases = phases
        self.ratio = ratio
        self.cur_phase = 0
        self.sv_best_model = sv_best_model
        sv_best_model.path.mkdir(exist_ok=True)
        sv_best_model.path = sv_best_model.path/f'phase{self.cur_phase}'
        super().__init__(layer_opt, wd, wd_ratio)

    def on_batch_begin(self):
        super().on_batch_begin()
        if self.iteration == len(self.phases[self.cur_phase]):
            self.cur_phase += 1
            self.iteration = 0
            self.sv_best_model.path = self.sv_best_model.path.parent/f'phase{self.cur_phase}'
            self.sv_best_model.best_metric = None
        self.layer_opt.set_lrs(ratio_listify(self.phases[self.cur_phase][self.iteration], self.ratio))


class SaveBestModel(CallBack):
    def __init__(self, learner, small_better, path='model/best'):
        self.path = Path(path)
        self.best_metric = None
        self.learner = learner
        self.small_better = small_better

    def on_epoch_end(self, trn_loss, vals):
        if not vals: return
        metric = vals[-1]
        if self.small_better:
            metric = -metric
        if not self.best_metric or metric >= self.best_metric:
            self.best_metric = metric
            self.learner.save(self.path)

    def on_train_end(self):
        best = -self.best_metric if self.small_better else self.best_metric
        print(f'best metric: {best:.6f}')
