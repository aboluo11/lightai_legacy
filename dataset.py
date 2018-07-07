from .imps import *

class MixDataset:
    def __init__(self, df, cat_cols, cont_cols):
        self.cat, self.cont, self.text = df[cat_cols].values,df[cont_cols].values,df['text'].values
        self.y = df['deal_probability'].values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idxs):
        return [self.cat[idxs], self.cont[idxs], self.text[idxs], self.y[idxs]]

class TabularDataset:
    def __init__(self, df, cat_cols, cont_cols):
        self.cat, self.cont = df[cat_cols].values,df[cont_cols].values
        self.y = df['deal_probability'].values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idxs):
        return [self.cat[idxs], self.cont[idxs], self.y[idxs]]

class TextDataset:
    def __init__(self, df):
        self.text = df['text'].values
        self.y = df['deal_probability'].values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idxs):
        return [self.text[idxs], self.y[idxs]]