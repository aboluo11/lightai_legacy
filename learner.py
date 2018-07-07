from .imps import *
from .callback import *
from .layer_optimizer import *

class Learner:
    def __init__(self, data, model, crit, metrics=None, opt_fn=torch.optim.Adam, path='./model'):
        self.data,self.model,self.crit = data,model,crit
        self.metrics = [] if metrics is None else metrics
        self.model_path = Path(path)
        self.model_path.mkdir(exist_ok=True)
        layer_groups = self.model.get_layer_groups()
        self.layer_opt = LayerOptimizer(layer_groups, opt_fn)
        self.recorder = Recorder(self.layer_opt)
        self.callbacks = [self.recorder]

    def fit(self, n_epochs=1, lrs=None, wds=None, clr_params=None, lr_find_params=None):
        if lrs is not None: lrs = np.array(listify(lrs, self.layer_opt.layer_groups))
        if wds is not None: wds = np.array(listify(wds, self.layer_opt.layer_groups))
        if lr_find_params is not None:
            sched = LR_Finder(lrs,lr_find_params[1],wds,nb=len(self.data.trn_dl),layer_opt=self.layer_opt)
            callbacks = [lr_find_params[0], sched]
        else:
            if clr_params is not None:
                v_ratio,h_ratio,tl_v_pct,tl_h_pct = clr_params
                sched = CircularLR(self.layer_opt,lrs,wds,v_ratio,h_ratio,len(self.data.trn_dl)*n_epochs,
                tl_v_pct=tl_v_pct,tl_h_pct=tl_h_pct)
            else:
                sched = ConstantLR(lrs,wds,self.layer_opt)
            callbacks = self.callbacks + [sched]

        for cb in callbacks: cb.on_train_begin()

        avg_mom,avg_loss,batch_num = 0.98,0,0
        names = ["epoch", "trn_loss", "val_loss"] + [m.__name__.lower() for m in self.metrics]
        layout = "{:^11}" * len(names)
        for epoch in tnrange(n_epochs, desc='Epoch'):
            self.model.train()
            num_batch = len(self.data.trn_dl)
            t = tqdm(self.data.trn_dl, leave=False, total=num_batch, ncols=125)
            try:
                for (*x,y) in t:
                    for cb in callbacks: cb.on_batch_begin()
                    batch_num += 1
                    loss = self.step(x,y)
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

            val_res = self.eval()
            for cb in callbacks: cb.on_epoch_end(debias_loss, val_res)

            if epoch == 0: print(layout.format(*names))
            self.print_stats(epoch+1, [debias_loss] + val_res)

    def eval(self):
        losses,bses = [],[]
        metrics = [m() for m in self.metrics]
        self.model.eval()
        with torch.no_grad():
            for (*x,y) in self.data.val_dl:
                predict = self.model(x).view(-1)
                loss = self.crit(predict, y)
                losses.append(loss.item())
                bses.append(len(y))
                for m in metrics:
                    m(predict, y)
        return [np.average(losses,weights=bses)] + [m.res(bses) for m in metrics]

    def step(self, x, y):
        """x: list"""
        predict = self.model(x).view(-1)
        self.layer_opt.opt.zero_grad()
        loss = self.crit(predict, y)
        loss.backward()
        self.layer_opt.opt.step()
        return loss.item()

    def print_stats(self, epoch, values):
        layout = "{:^11}" + "{:^11.6f}" * len(values)
        print(layout.format(epoch, *values))

    def lr_find(self, start_lrs=[1e-5], end_lr=20, wds=[0]):
        self.save(self.model_path/'tmp.h5')
        recorder = Recorder(self.layer_opt)
        self.fit(lrs=start_lrs,wds=wds,lr_find_params=[recorder,end_lr])
        recorder.plot_lr_loss()
        self.load(self.model_path/'tmp.h5')

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))