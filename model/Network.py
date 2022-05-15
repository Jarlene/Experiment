from model.BaseModel import Base
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch.nn as nn
import torch


class Network(Base):
    def __init__(self, resnet, args) -> None:
        super(Network, self).__init__()
        self.args = args
        self.resnet = resnet
        self.feature_dim = args.feature_dim
        self.cluster_num = args.class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, batch):

        h_i = self.resnet(batch['x_i'])
        h_j = self.resnet(batch['x_j'])

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def loss(self, batch):
        z_i, z_j, c_i, c_j = self.forward(batch)
        loss = self.compute_cl_loss(z_i, z_j, 
                                    temperature=self.args.temperature, 
                                    tau_plus=self.args.tau_plus, 
                                    debiased=self.args.debiased)
        if self.args.use_kl_loss:
            loss += self.compute_kl_loss(c_i, c_j)

        return loss

