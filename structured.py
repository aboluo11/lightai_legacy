from .imps import *
from pandas.api.types import is_string_dtype, is_numeric_dtype

def prepro(train, valid, min_freq):
    to_cat(train, valid, min_freq)
    numericalize(train, valid)
    fix_missing(train, valid)

def to_cat(train, valid, min_freq):
    for col in train.columns:
        if is_string_dtype(train[col]):
            r = train[col].value_counts() < min_freq
            r = r[r].to_frame()
            train.loc[train[col].to_frame().merge(r,left_on=col,right_index=True).index,col] = '__LessThan__'
            train[col] = train[col].astype('category')
            valid[col] = pd.Categorical(valid[col], categories=train[col].cat.categories)

def numericalize(train, valid):
    for col in train.columns:
        if train[col].dtype.name == 'category':
            lt = (train[col]=='__LessThan__')
            train[col] = train[col].cat.codes+1
            valid[col] = valid[col].cat.codes+1
            train.loc[lt, col] = 0

def fix_missing(train, valid):
    for col in train.columns:
        if is_numeric_dtype(train[col]):
            if pd.isna(train[col]).sum():
                train[f'{col}_na'] = pd.isna(train[col])
                mean = train[col].mean()
                train[col] = train[col].fillna(mean)
                valid[f'{col}_na'] = pd.isna(valid[col])
                valid[col] = valid[col].fillna(mean)

def train_valid_split(raw, persentage):
    nrows = len(raw)
    idx = np.arange(nrows)
    np.random.shuffle(idx)
    train, valid = raw.iloc[idx[:int(nrows*persentage)]], raw.iloc[idx[int(nrows*persentage):]]
    return train.copy(), valid.copy()

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)