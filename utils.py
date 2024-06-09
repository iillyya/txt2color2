import numpy as np
import torch
import torch.nn as nn
import warnings
from skimage.color import lab2rgb, rgb2lab

# ======================== For text embeddings ======================== #
SOS_token = 0
EOS_token = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        self.max_len = 0

    def index_elements(self, data):
        for element in data:
            self.max_len = len(data) if self.max_len < len(data) else self.max_len
            self.index_element(element)

    def index_element(self, element):
        if element not in self.word2index:
            self.word2index[element] = self.n_words
            self.word2count[element] = 1
            self.index2word[self.n_words] = element
            self.n_words += 1
        else:
            self.word2count[element] += 1

def load_pretrained_embedding(dictionary, embed_file, embed_dim):
    if embed_file is None: return None

    pretrained_embed = {}
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            if word == '<unk>':
                continue
            pretrained_embed[word] = entries
        f.close()

    vocab_size = len(dictionary) + 2
    W_emb = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for word, index in dictionary.items():
        if word in pretrained_embed:
            W_emb[index, :] = pretrained_embed[word]
            n += 1

    return W_emb

class Embed(nn.Module):
    def __init__(self, vocab_size, embed_dim, W_emb, train_emb):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        if W_emb is not None:
            self.embed.weight = nn.Parameter(W_emb)

        if train_emb == False:
            self.embed.requires_grad = False

    def forward(self, doc):
        doc = self.embed(doc)
        return doc


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def lab2rgb_1d(in_lab, clip=True):
    warnings.filterwarnings("ignore")
    tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    return tmp_rgb

def init_weights_normal(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0.0, 0.05)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.05)
