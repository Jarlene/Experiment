


import torch
from model.BaseModel import Base
from model.layer import FeaturesEmbedding,InteractionLayer, FusionLayer

class AOA(Base):

    def __init__(self, args) -> None:
        super(AOA, self).__init__()
        self.args = args
        self.embed = FeaturesEmbedding(args.field_dims, args.emb_size)
        self.interact_layer = InteractionLayer(field_nums=args.feature_nums, use_atten=True, emb_size=args.emb_size)
        self.fusion_layer = FusionLayer(field_nums=args.feature_nums)
        size = args.emb_size 
        self.seq = torch.nn.Sequential()
        for idx, hide in enumerate(args.hide_nums):
            self.seq.add_module(name='linear_'+str(idx), module=torch.nn.Linear(size, hide))
            size = hide
        self.classifer = torch.nn.Linear(size, args.num_class)


    def forward(self, batch): 
        x = batch['x']
        x = self.embed(x)
        x1 = x[..., :self.args.feature_nums[0], :]
        x2 = x[..., self.args.feature_nums[0]:, :]
        out = self.interact_layer(x1, x2)
        out = self.fusion_layer(out)
        out = torch.flatten(out, start_dim=1)
        out = self.seq(out)
        return self.classifer(out)


    def loss(self, batch):
        res = self.forward(batch=batch)
        y = batch[self.args.label]
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(res, y)
        return loss
