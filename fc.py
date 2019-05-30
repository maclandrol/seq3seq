from torch.nn import Linear, Sequential, ReLU, Dropout
from commons import UnitNormLayer, ClonableModule

class FCLayer(ClonableModule):
    def __init__(self, input_size, hidden_dims, normalize_features=True, dropout=0.):
        super(FCLayer, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.normalize_kernel = normalize_features
        self.dropout = dropout
        layers = []
        in_ = input_size
        for i, out_ in enumerate(hidden_dims):
            layers.append(Linear(in_, out_))
            layers.append(ReLU())
            layers.append(Dropout(self.dropout))
            if normalize_features:
                layers.append(UnitNormLayer())
            in_ = out_
            
        self.net = Sequential(*layers)

    @property
    def output_dim(self):
        return self.hidden_dims[-1] if len(self.hidden_dims) > 0 else self.input_size

    def forward(self, x):
        return self.net(x)

    def clone(self):
        model = FCLayer(**self.params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model