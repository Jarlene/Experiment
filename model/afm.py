from model.BaseModel import Base
from model.layer import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine

import torch.nn.functional as F
import torch


class AFM(Base):
    def __init__(self, args):
        super(AFM, self).__init__()
        self.args = args
        self.num_fields = len(args.field_dims)
        self.embedding = FeaturesEmbedding(args.field_dims, args.embed_dim)
        self.linear = FeaturesLinear(args.field_dims)
        self.afm = AttentionalFactorizationMachine(args.embed_dim, args.attn_size, args.dropouts)
        self.criterion = torch.nn.BCELoss()

    def forward(self, batch):
        x = batch['x']
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


    def loss(self, batch):
        pred = self.forward(batch)
        targt =  batch[self.args.label]
        loss = self.criterion(pred, targt.float())
        return loss




