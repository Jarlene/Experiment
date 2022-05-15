
import torch
from experiment import Experiment, AutoDateSet, train, get_train_args
from model.aoa import AOA

from dataset.criteo import CriteoDataset


def get_model(args):
    model = AOA(args)
    return model

def get_dataset(args):
    dataset = CriteoDataset(dataset_path=args.dataset_paths[0], cache_path='data/.criteo')
    train_length = int(len(dataset) * 0.9)
    valid_length = len(dataset) - train_length
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (train_length, valid_length))
    test_dataset = CriteoDataset(dataset_path=args.dataset_paths[1], cache_path='data/.criteo')
    return [train_dataset, valid_dataset, test_dataset], dataset.field_dims

def main():
    args = get_train_args()
    args.hide_nums= [256,128,64]
    args.emb_size = 16
    args.label='y'
    args.num_class = 2
    args.dataset_paths= ['data/criteo/train.txt', 'data/criteo/test.txt']
    datasets, field_dims= get_dataset(args)
    args.field_dims = field_dims
    args.feature_nums = [datasets[2].NUM_INT_FEATS, datasets[2].NUM_FEATS-datasets[2].NUM_INT_FEATS]
    model = get_model(args)
    
    experiment = Experiment(model=model, args=args)
    # dummy_data = torch.randint(0, 26,(1, len(args.field_dims)))
    # experiment.configure_example_input({'x':dummy_data})
    data = AutoDateSet(datasets, args.batch_size, args.batch_size, args.num_workers, args.pin_memory)
    train(args, experiment, data)

if __name__ == "__main__":
    main()