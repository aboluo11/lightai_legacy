from .imps import *
from .callback import *
from .layer_optimizer import *


class Learner:
    def __init__(self, trn_dl, val_dl, model, crit, layer_opt, metric=None, small_better=True,
                 sv_best_path='./model/best', writer=None):
        self.trn_dl, self.val_dl, self.model, self.crit, self.metric = trn_dl, val_dl, model, crit, metric
        self.layer_opt = layer_opt
        self.recorder = Recorder(self.layer_opt, writer)
        self.sv_best_model = SaveBestModel(self, small_better, path=sv_best_path)
        self.callbacks = [self.recorder, self.sv_best_model]
        self.global_step = 0
        self.epoch = 0
        self.writer = writer

    def fit(self, phases, mode, ratio=None, wd=None, wd_ratio=None, print_stats=True, addtional_cbs=None):
        if not ratio:
            ratio = [1] * len(self.layer_opt)
        if not wd_ratio:
            wd_ratio = [1] * len(self.layer_opt)
        ratio = np.array(ratio)
        wd_ratio = np.array(wd_ratio)
        n_batches = sum([len(phase) for phase in phases])
        n_epochs = n_batches // len(self.trn_dl)
        print(f'total: {n_epochs} epochs')
        if mode is None:
            callbacks = self.callbacks
        elif mode == 'cyclic':
            sched = Cyclic(phases, ratio, wd, wd_ratio, self.layer_opt, self.sv_best_model)
            callbacks = self.callbacks + [sched]
        elif mode == 'phase':
            sched = PhaseLr(phases, ratio, wd, wd_ratio, self.layer_opt)
            callbacks = self.callbacks + [sched]
        elif mode == 'lr_find':
            sched = LRFinder(phases, ratio, wd, wd_ratio, self.layer_opt)
            self.recorder = Recorder(self.layer_opt, None)
            callbacks = [sched, self.recorder]
        if addtional_cbs:
            callbacks += addtional_cbs
        for cb in callbacks:
            cb.on_train_begin()
        avg_mom, avg_loss, batch_num = 0.98, 0, 0
        names = ["epoch", "trn_loss"] + (["val_loss"] if self.val_dl else []) + \
                ([self.metric.__class__.__name__.lower()] if self.metric else []) + ["time"]
        layout = "{:^11}" * len(names)
        for epoch in range(n_epochs):
            time1 = time.time()
            self.train()
            for [x, target] in self.trn_dl:
                x, target = T(x), T(target)
                for cb in callbacks:
                    cb.on_batch_begin()
                batch_num += 1
                loss = self.step(x, target)
                avg_loss = avg_loss * avg_mom + loss * (1 - avg_mom)
                debias_loss = avg_loss / (1 - avg_mom ** batch_num)
                stop = False
                for cb in callbacks:
                    stop = stop or cb.on_batch_end(debias_loss, self.model)
                if stop:
                    return
                self.global_step += 1
            if self.val_dl and mode != 'lr_find':
                val_res = self.eval()
            else:
                val_res = None
            for cb in callbacks:
                cb.on_epoch_end(debias_loss, val_res, self.epoch)
            if print_stats:
                if epoch == 0:
                    print(layout.format(*names))
                self.print_stats(epoch + 1, debias_loss, *(val_res if val_res else []), time.time()-time1)
            self.epoch += 1
        for cb in callbacks:
            cb.on_train_end()

    def eval(self):
        losses, bses = [], []
        self.model.eval()
        with torch.no_grad():
            for tta_batch in self.val_dl:
                predicts = []
                for i, (x, target) in enumerate(tta_batch):
                    x, target = T(x), T(target)
                    predict = self.model(x)
                    predicts.append(predict)
                    if i == 0:
                        loss = self.crit(predict, target)
                        losses.append(loss.item())
                        bses.append(len(target))
                if self.metric:
                    self.metric(predicts, target)
        loss = np.average(losses, weights=bses)
        if self.metric:
            return [loss, self.metric.res(self.epoch)]
        else:
            return [loss]

    def step(self, x, target):
        predict = self.model(x, self.global_step)
        self.layer_opt.opt.zero_grad()
        loss = self.crit(predict, target)
        loss.backward()
        self.layer_opt.opt.step()
        return loss.item()

    def lr_find(self, start_lr=1e-5, end_lr=20, ratio=None, wd=None, wd_ratio=None, n_epochs=1):
        self.save('model/tmp')
        phases = [np.geomspace(start_lr, end_lr, num=n_epochs*len(self.trn_dl), endpoint=True)]
        saved_recorder = self.recorder
        self.fit(phases, mode='lr_find', ratio=ratio, wd=wd, wd_ratio=wd_ratio)
        self.recorder.plot_lr_loss()
        self.recorder = saved_recorder
        self.load('model/tmp')

    def train(self):
        def f(m):
            if getattr(m, 'train_mode', True):
                m.train()
            else:
                m.eval()

        for lg in self.layer_opt.layer_groups:
            for m in lg:
                f(m)

    def freeze_to(self, right, bn_freeze):
        for lg in self.layer_opt.layer_groups[:right]:
            set_lg_train_mode(lg, not bn_freeze)
            for p in self.layer_opt.lg_params(lg):
                p.requires_grad = False
        for lg in self.layer_opt.layer_groups[right:]:
            set_lg_train_mode(lg, True)
            for p in self.layer_opt.lg_params(lg):
                p.requires_grad = True

    def unfreeze(self):
        for lg in self.layer_opt.layer_groups:
            set_lg_train_mode(lg, True)
            for p in self.layer_opt.lg_params(lg):
                p.requires_grad = True

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.layer_opt.opt.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.layer_opt.opt.load_state_dict(checkpoint['optimizer'])

    def print_stats(self, *values):
        layout = "{:^11}" + "{:^11.6f}" * (len(values) - 1)
        print(layout.format(*values))


def set_lg_train_mode(lg, train):
    for m in lg:
        m.train_mode = train
