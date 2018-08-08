from .imps import *

class RNNEncoder(nn.Module):
    def __init__(self,emb_sz,hid_sz,layers,num_words,bidirectional,rnn_dp):
        super().__init__()
        self.emb = nn.Embedding(num_words,emb_sz, padding_idx=1)
        self.rnn = nn.GRU(emb_sz,hid_sz,layers,batch_first=True,bidirectional=bidirectional,dropout=rnn_dp)
        
    def forward(self, x):
        x = self.emb(x)
        if self.hidden is None:
            outp, hidden = self.rnn(x)
        else:
            outp, hidden = self.rnn(x,self.hidden)
        self.hidden = hidden.detach()
        return outp

class MultiBatchRNN(nn.Module):
    def __init__(self,bptt,n_bptt,emb_sz,hid_sz,layers,num_words,bidirectional,rnn_dp):
        super().__init__()
        self.rnn = RNNEncoder(emb_sz, hid_sz, layers, num_words,bidirectional,rnn_dp)
        self.bptt,self.n_bptt,self.hid_sz = bptt,n_bptt,hid_sz
        
    def pool(self,x,is_max):
        bs = x.size()[0]
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(0,2,1),1).view(bs,-1)
            
    def forward(self,x):
        bs,sl = x.size()
        self.rnn.hidden = None
        res = []
        for i in range(0,sl,self.bptt):
            outp = self.rnn(x[:, i:min(i+self.bptt,sl)])
            # if i+1 > sl - self.bptt*self.n_bptt:
            #     res.append(outp)
            res.append(outp)
        res = torch.cat(res,1)
        res = torch.cat([outp[:,-1],self.pool(res,True),self.pool(res,False)], 1)
        return res

class LinerBlock(nn.Module):
    def __init__(self, ni, no, drop):
        super().__init__()
        self.lin = nn.Linear(ni, no, bias=False)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)
        
    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))

class Tabular(nn.Module):
    def __init__(self, emb_szs, units, drops, cont_sz):
        super().__init__()
        units.insert(0, sum([sz[1] for sz in emb_szs]) + cont_sz)
        self.embs = nn.ModuleList([nn.Embedding(ni, no) for ni,no in emb_szs])
        self.lins = nn.ModuleList([LinerBlock(units[i], units[i+1], drops[i]) for i in range(len(drops))])

    def forward(self, x):
        cat, cont = x
        cat = torch.cat([e(cat[:,i]) for i,e in enumerate(self.embs)], 1)
        x = torch.cat([cat, cont], 1)
        for l in self.lins:
            raw_x = l(x)
            x = F.relu(raw_x)
        return raw_x

    def get_layer_groups(self):
        return [self.embs, self.lins]

class Text(nn.Module):
    def __init__(self, bptt, n_bptt, units, layers, num_words, wd_emb_sz, hid_sz, drops, rnn_dp,
     bidirectional=True):
        super().__init__()
        units.insert(0, 3*hid_sz*(2 if bidirectional else 1))
        self.rnn = MultiBatchRNN(bptt=bptt, n_bptt=n_bptt, emb_sz=wd_emb_sz, hid_sz=hid_sz, layers=layers,
         num_words=num_words, bidirectional=bidirectional, rnn_dp=rnn_dp)
        self.lins = nn.ModuleList([LinerBlock(units[i], units[i+1], drops[i]) for i in range(len(drops))])

    def forward(self, text):
        x = self.rnn(text[0])
        for l in self.lins:
            raw_x = l(x)
            x = F.relu(raw_x)
        return raw_x