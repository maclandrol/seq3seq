import random
import torch
import torch.nn.utils.rnn as rnn_utils

from commons import *
from const import PADVAL, VOCAB
from fc import FCLayer
from rnn import RNNDecoder, RNNEncoder
#from cnn import CNNDecoder, CNNEncoder
from torch.nn import Embedding
from torch.nn import GRU
from torch.nn import LSTM
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import functional as F

EOS_IDX = len(VOCAB)  # padding value is 0

def loss_fn(ypred, y, w=None, contrib=1.0):
    ypred, (x, xpred) = ypred
    xpred = xpred.transpose(0,1)[1:].view(-1, xpred.shape[-1])
    x = x.transpose(0,1)[1:].contiguous().view(-1)
    sup_loss = F.binary_cross_entropy_with_logits(ypred, y, weight=w)
    gen_loss = F.cross_entropy(xpred, x, ignore_index=PADVAL)
    return gen_loss + contrib*sup_loss


class Seq3Seq(Module):
    def __init__(self, vocab_size, embedding_size, output_size, config, net_type="rnn"):
        super(Seq3Seq, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        # using a shared embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size) # EmbeddingViaLinear
        if net_type != "rnn":
            raise NotImplementedError("Only rnn type have been implemented")
        self.encoder = RNNEncoder(self.embedding, **config["encoder"])
        self.decoder = RNNDecoder(self.embedding, **config["decoder"])


        self.latent_dim = self.encoder.output_dim
        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"
        supervised_net = FCLayer(self.encoder.output_dim, **config["supervised"])
        self.supervised = nn.Sequential(supervised_net, Linear(supervised_net.output_dim, self.output_size))
    
    def encoder_forward(self, inp):
        return self.encoder(inp)

    def decoder_forward(self, inp, hidden):
        return self.decoder(inp, hidden)

    def supervised_forward(self, inp):
        return self.supervised(inp)

    def get_fingerprint(self, x):
        _, hidden = self.encoder_forward(x)
        return self._fingerprint(hidden, x.size(0))

    def _fingerprint(self, hidden, batch_size):
        if isinstance(hidden, tuple):
            hidden = hidden[0] # discard cell state
        return hidden.transpose(0, 1).contiguous().view(batch_size, -1)

    def forward(self, x, teacher_forcing_ratio=0.75):
        # x is source and target: bsize, vocab
        batch_size = x.size(0)
        max_len = x.size(1)
        _, hidden = self.encoder_forward(x)
        fp = self._fingerprint(hidden, batch_size)
        pred = self.supervised_forward(fp)

        outputs = x.new_zeros((max_len, batch_size, self.vocab_size)).float()
        dec_output = x[:, 0] # should always be <bos>
        for i in range(1, max_len):
            dec_output, hidden = self.decoder_forward(dec_output, hidden)
            outputs[i] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            symbols = dec_output.max(1)[1] # max is still differentiable    
            dec_output = x[:, i] if teacher_force else symbols
            # if i <5:
            #     print(dec_output, x[:, i], symbols, "\n")
                # eventually do something if eos are returned prematurely
            if torch.any(symbols.data.eq(EOS_IDX)):
                pass
        return pred, (x, outputs.transpose(0,1))