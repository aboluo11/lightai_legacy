from .imps import *

class Tokenizer:
    def __init__(self, num_words=60000, min_freq=2):
        self.num_words = num_words
        self.min_freq = min_freq
        self.sp = re.compile(r'([!"#$%&()*+,.\-/:;<=>?@[\\\]^`{|}~\t\n ])')

    def fit(self, texts):
        word_counts = collections.defaultdict(lambda:0)
        for text in texts:
            text = text.lower()
            it = filter(lambda x:bool(x) and x != ' ', self.sp.split(text))
            for word in it:
                word_counts[word] += 1
        freq = collections.Counter(word_counts)
        self.stoi = {w:i+2 for i,(w,c) in enumerate(freq.most_common(self.num_words)) if c >= self.min_freq}

    def numberize(self, texts):
        """unknow: 0, pad: 1"""
        res = []
        for text in texts:
            text = text.lower()
            it = filter(lambda x:bool(x) and x != ' ', self.sp.split(text))
            seq = [self.stoi[w] if w in self.stoi else 0 for w in it]
            res.append(seq)
        return res

def pad_seqs(seqs):
    lens = [len(seq) for seq in seqs]
    max_len = max(lens)
    res = np.ones([len(seqs), max_len],dtype=np.int64)
    for i,seq in enumerate(seqs):
        res[i, -len(seq):] = seq
    return res