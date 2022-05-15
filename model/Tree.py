import torch
from model.BaseModel import Base
from model.layer import FeaturesEmbedding

class Tree(Base):

    def __init__(self, args) -> None:
        super(Tree, self).__init__()
        input_size = args.embed_dim * len(args.field_dims)
        self.args = args
        self.mlp = torch.nn.ModuleList()
        self.classifers = torch.nn.ModuleList()
        self.embedding = FeaturesEmbedding(args.field_dims, args.embed_dim)
        for i, h in enumerate(args.hidden):
            self.mlp.append(torch.nn.Linear(input_size, h))
            self.classifers.append(torch.nn.Linear(h, args.num_class))
            input_size = h

        if args.activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif args.activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, batch):
        x = batch['x']
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        res = []
        for i, layer in enumerate(self.mlp):
            x =  self.activation(layer(x))
            y = self.classifers[i](x)
            res.append(y)
        return res

    def loss(self, batch):
        res = self.forward(batch)
        y = batch[self.args.label]
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = 0
        for i, r in enumerate(res):
            loss += loss_fun(r, y)
        return loss



