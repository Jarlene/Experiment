
import torch
from experiment import Experiment, AutoDateSet, train, get_args
from model.Tree import Tree

from dataset.criteo import CriteoDataset


def get_model(args):
    model = Tree(args)
    return model

def get_dataset(args):
    dataset = CriteoDataset(dataset_path=args.dataset_paths[0])
    train_length = int(len(dataset) * 0.9)
    valid_length = len(dataset) - train_length
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (train_length, valid_length))
    test_dataset = CriteoDataset(dataset_path=args.dataset_paths[1])
    return [train_dataset, valid_dataset, test_dataset], dataset.field_dims

def main():
    args = get_args()
    args.hidden= [1024,512,256,128,64]
    args.embed_dim = 16
    args.activation='relu'
    args.label='y'
    args.num_class = 2
    args.dataset_paths= ['data/criteo/train.txt', 'data/criteo/test.txt']
    datasets, field_dims= get_dataset(args)
    print(field_dims)
    args.field_dims = field_dims
    model = get_model(args)
    
    experiment = Experiment(model=model, args=args)
    data = AutoDateSet(datasets, args.batch_size, args.batch_size, args.num_workers, args.pin_memory)
    train(args, experiment, data)

if __name__ == "__main__":
    main()