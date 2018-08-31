from .imps import *
from .callback import *
from .layer_optimizer import *

class Learner:
    def __init__(self, trn_dl, val_dl, model, crit,layer_opt, metric=None, small_better=True,
                 sv_best_path='./model/best'):
        self.trn_dl,self.val_dl,self.model,self.crit,self.metric = trn_dl,val_dl,model,crit,metric
        self.layer_opt = layer_opt
        self.recorder = Recorder(self.layer_opt)
        self.callbacks = [self.recorder, SaveBestModel(self, small_better, path=sv_best_path)]

    def fit(self, n_epochs, lrs, wds=None, clr_params=None, callbacks=None, print_stats=True):
        if callbacks is None:
            if clr_params is not None:
                v_ratio,h_ratio,tl_v_pct,tl_h_pct = clr_params
                sched = CircularLR(self.layer_opt,lrs,wds,v_ratio,h_ratio,len(self.trn_dl)*n_epochs,
                tl_v_pct=tl_v_pct,tl_h_pct=tl_h_pct)
            else:
                sched = ConstantLR(lrs,wds,self.layer_opt)
            callbacks = self.callbacks + [sched]

        for cb in callbacks: cb.on_train_begin()

        avg_mom,avg_loss,batch_num = 0.98,0,0
        names = ["epoch", "trn_loss"] + (["val_loss"] if self.val_dl else []) +\
         ([self.metric.__name__.lower()] if self.metric else [])
        layout = "{:^11}" * len(names)
        for epoch in tnrange(n_epochs, desc='Epoch', ascii=True):
            self.train()
            t = tqdm(self.trn_dl, leave=False, ascii=True, ncols=125)
            try:
                for [x, target] in t:
                    x, target = T(x), T(target)
                    for cb in callbacks: cb.on_batch_begin()
                    batch_num += 1
                    loss = self.step(x, target)
                    avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
                    debias_loss = avg_loss / (1 - avg_mom**batch_num)
                    t.set_postfix(loss=debias_loss)
                    stop = False
                    for cb in callbacks:
                        stop = stop or cb.on_batch_end(debias_loss)
                    if stop: return
            finally:
                t.leave = True
                t.close()
            if self.val_dl: val_res = self.eval()
            else: val_res = None
            for cb in callbacks: cb.on_epoch_end(debias_loss, val_res)
            
            if print_stats:
                if epoch == 0: print(layout.format(*names))
                self.print_stats(epoch+1, [debias_loss] + (val_res if val_res else []))
        for cb in callbacks: cb.on_train_end()

    def eval(self):
        losses,bses = [],[]
        metric = self.metric() if self.metric else None
        self.model.eval()
        with torch.no_grad():
            for x, target in self.val_dl:
                x, target = T(x), T(target)
                predict = self.model(x)
                loss = self.crit(predict, target)
                losses.append(loss.item())
                bses.append(len(target))
                if metric:
                    metric(predict, target)
        loss = np.average(losses,weights=bses)
        if metric: return [loss,metric.res()]
        else: return [loss]

    def step(self, x, target):
        predict = self.model(x)
        self.layer_opt.opt.zero_grad()
        loss = self.crit(predict, target)
        loss.backward()
        self.layer_opt.opt.step()
        return loss.item()

    def print_stats(self, epoch, values):
        layout = "{:^11}" + "{:^11.6f}" * len(values)
        print(layout.format(epoch, *values))

    def lr_find(self, start_lrs=None, end_lrs=None, wds=None, n_epochs=1):
        if start_lrs is None: start_lrs = [1e-5]
        if end_lrs is None: end_lrs = [20]
        self.save('model/tmp')
        recorder = Recorder(self.layer_opt)
        lr_finder = LR_Finder(
            start_lrs,end_lrs,wds,nb=n_epochs*len(self.trn_dl),layer_opt=self.layer_opt)
        self.fit(n_epochs,lrs=start_lrs,wds=wds,callbacks=[recorder,lr_finder])
        recorder.plot_lr_loss()
        self.load('model/tmp')

    def train(self):
        def f(m):
            if getattr(m, 'train_mode', True):
                m.train()
            else: m.eval()
        for lg in self.layer_opt.layer_groups:
            for m in lg: f(m)

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
            'state_dict': self.model.state_dict(),
            'optimizer' : self.layer_opt.opt.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.layer_opt.opt.load_state_dict(checkpoint['optimizer'])