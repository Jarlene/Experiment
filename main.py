import torch
import torchvision
from experiment import Experiment, AutoDateSet, train, get_args
from model.Network import Network
from model.resnet import get_resnet
import model.transform as transform


def get_model(args):
    res = get_resnet(args.resnet)
    model = Network(res, args)
    return model

def get_dataset(args):
    train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
    test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
    return [train_dataset, test_dataset]


def collate_fn(batch):
    xi = []
    xj = []
    for b in batch:
        xi.append(b[0]['x_i'])
        xj.append(b[0]['x_j'])

    xi = torch.stack(xi)
    xj = torch.stack(xj)
    return {'x_i': xi, 'x_j': xj}

def main():
    args = get_args()
    args.dataset_dir = 'data/cifar100'
    args.resnet = 'ResNet34'
    args.feature_dim = 128
    args.class_num = 20
    args.batch_size = 128
    args.image_size = 224

    model = get_model(args)
    experiment = Experiment(model=model, args=args)
    datasets = get_dataset(args)
    data = AutoDateSet(datasets, args.batch_size, args.batch_size, args.num_workers, args.pin_memory, collate_fn=collate_fn)
    train(args, experiment, data)


if __name__ == "__main__":
    main()