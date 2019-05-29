import numpy as np
import os
import pandas as pd
import sys
import torch
import yaml

from functools import partial
from ivbase.transformers.data import WeightBalancing
from ivbase.utils.commons import to_tensor, is_dtype_numpy_array, is_dtype_torch_tensor
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from . import const

class SeqTransformer:

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.alphabet2int = {el: i + 1 for i, el in enumerate(alphabet)}

    def _transform(self, seq):
        seq = const.BOS + seq + const.EOS # adding <bos> and <eos>
        assert isinstance(
            seq, str), "SequenceTransformer expect a smiles (string)"
        i, l, result = 0, 0, [const.PADVAL for _ in range(len(seq))]
        while i < len(seq):
            # Is two-letter symbol?
            if self.alphabet2int.get(seq[i:i + 2]):
                result[l] = self.alphabet2int.get(seq[i:i + 2], 0)
                i += 2
            else:
                result[l] = self.alphabet2int.get(seq[i], 0)
                i += 1
            if result[l] == 0:
                warnings.warn('Some out of vocabulary encounter during the transformation.'
                              'Please make sure you vocabulary is exhaustive.', RuntimeWarning)
            l += 1
        return np.asarray(result[:l])

    def transform(self, sequences):
        return [self._transform(seq) for seq in sequences]

    def __call__(self, sequences):
        vec_seq = self.transform(sequences)
        return vec_seq

class SeqDataset(Dataset):
    def __init__(self, X, y, w=None, cuda=False):
        self.smiles, self.X, self.y, self.w = transform_data(X, y, w)
        self.cuda = cuda

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.w is not None:
            return to_tensor(self.X[idx], gpu=self.cuda, dtype=torch.long), to_tensor(self.y[idx, :], gpu=self.cuda), to_tensor(self.w[idx, :], gpu=self.cuda, dtype=torch.float)
        return to_tensor(self.X[idx], gpu=self.cuda, dtype=torch.long), to_tensor(self.y[idx, :], gpu=self.cuda)

def collate_func(batch):
    x, *y = zip(*batch)
    x = pad_sequence(x, batch_first=True)
    x_ = x.view(x.size(0), -1)
    lengths = (x_ > 0).long().sum(1)
    lengths, perm_idx = lengths.sort(0, descending=True)
    x = x[perm_idx]
    y = [torch.stack(yy)[perm_idx] for yy in y]
    return (x, *y)
    

def tensor2string(tensor, rem_bos=True, rem_eos=True):
    ids = torch.argmax(tensor, dim=-1)
    if len(ids) == 0:
        return ''
    string = ''.join([const.VOCAB[id-1]if id>0 else "" for id in ids])
    if rem_bos and string[0] == const.BOS:
        string = string[1:]

    if rem_eos and string[-1] == const.EOS:
        string = string.split(const.EOS)[0]
    return string
              

def read_data(file_name, num_columns=-2, max_n=-1, has_header=True):
    if has_header:
        datafile = pd.read_csv(file_name, header=0, engine='python')
    else:
        datafile = pd.read_csv(file_name, header=None, engine='python')
    datafile.fillna(0, inplace=True)  # Add a 0 where data is missing
    data = datafile.values
    smiles = data[:, -1]  # The last column is the smiles
    y = data[:, 0:num_columns]
    return smiles[:max_n], y[:max_n]

def transform_data(smiles, y, w=None):
    # Initialize the transformer
    trans = SeqTransformer(alphabet=const.VOCAB)
    X = trans(smiles)
    y = y.astype(np.float32)
    return smiles, X, y, w


def load_dataset(X, y, test_size=0.2, valid_size=0.25, shuffle=True, balance=False, **kwargs):
    # First separate the training from the testing set. Then repeat to separate the validation from the training set.
    if balance:
        wbalance = WeightBalancing(X, y)
        X, y, w = wbalance.transform(X, y)
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=test_size, shuffle=shuffle)
        if valid_size:
            x_train, x_valid, y_train, y_valid, w_train, w_valid= train_test_split(x_train, y_train, w_train, test_size=valid_size, shuffle=shuffle)
    else:
        w_train = w_test = w_valid = None
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, shuffle=shuffle)

    train_dt = SeqDataset(x_train, y_train, w=w_train, **kwargs)
    valid_dt = None if not valid_size else SeqDataset(x_valid, y_valid, w=w_valid, **kwargs)
    test_dt = SeqDataset(x_test, y_test, w=w_test, **kwargs)
    return train_dt, valid_dt, test_dt


def load_config(path):
    with open(path, 'r') as IN:
        model_params = yaml.safe_load(IN)
    return model_params