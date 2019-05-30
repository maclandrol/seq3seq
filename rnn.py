import torch.nn.utils.rnn as rnn_utils

from commons import *
from torch.nn import Embedding
from torch.nn import GRU
from torch.nn import LSTM
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential


class RNNEncoder(Module):
    def __init__(self, embedding_layer, hidden_dim, num_layers=1, 
                 bidirectional=True, dropout=0., activation="tanh",normalize_features=False, rnn_type="gru"):
        super(RNNEncoder, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        self.embedding_size = self.embedding.embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        self.activation = activation_map.get(activation)
        if self.rnn_type == "gru":
            rnn_class = GRU
        else:
            rnn_class = LSTM
        self.rnn = rnn_class(input_size=self.embedding_size,
                                        hidden_size=self.hidden_dim,
                                        num_layers=self.num_layers,
                                        bidirectional=self.bidirectional,
                                        batch_first=True,
                                        dropout=self.dropout)
        
        fp_layer = [nn.Linear(hidden_dim * (2 if self.bidirectional else 1), hidden_dim)]
        if self.activation is not None:
            fp_layer.append(self.activation)
        if normalize_features:
            fp_layer.append(UnitNormLayer())
        self.fc = Sequential(*fp_layer)
        self._output_dim = hidden_dim * num_layers

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        batch_size = x.size(0)
        embedding = self.embedding(x)
        lengths = (x.view(batch_size, -1) > 0).long().sum(1).data.cpu().numpy()
        # should not be needed if this if not variable length
        packed_x_train = rnn_utils.pack_padded_sequence(embedding, lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_x_train)
        # should not be needed if this if not variable length
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        hidden = hidden.view(self.num_layers, -1, batch_size, self.hidden_dim).transpose(1,2).contiguous()
        hidden = self.fc(hidden.view(self.num_layers, batch_size, -1))
        if self.rnn_type == "lstm":
            hidden = (hidden, torch.zeros_like(hidden))
        return output, hidden

    def clone(self):
        model = RNNLayer(**self.params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model

class RNNDecoder(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, num_layers=1, dropout=0., rnn_type="gru"):
        super(RNNDecoder, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.hidden_dim = hidden_dim
        self.embedding = embedding_layer
        self.embedding_size = self.embedding.embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "gru":
            rnn_class = GRU
        else:
            rnn_class = LSTM
        self.rnn = rnn_class(input_size=self.embedding_size,
                                        hidden_size=self.hidden_dim,
                                        num_layers=self.num_layers,
                                        dropout=self.dropout)
        
        self._output_dim = self.embedding.num_embeddings
        self.fc = nn.Linear(self.hidden_dim, self._output_dim)

    def forward(self, x, hidden):
        x = x.unsqueeze(0) # 1, bsize, indim
        embedding = self.embedding(x)
        output, hidden = self.rnn(embedding, hidden)
        output = output.squeeze(0) # bsize,  hid_dim
        output = self.fc(output) # bsize vocab
        return output, hidden
