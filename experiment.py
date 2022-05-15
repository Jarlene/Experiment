import os
from argparse import Namespace, _ArgumentGroup
import argparse
from tokenize import group
import numpy as np
from typing import List, Callable, Union, Any, Type, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from torch import optim
from model.BaseModel import Base
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchmetrics import AUROC, Accuracy, Precision, Recall, AUC, PrecisionRecallCurve
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.profiler import PyTorchProfiler

# METRIC_MAPPER = OrderedDict (
# [
#     ('accuracy', 'Accuracy'),
#     ('auroc', 'AUROC'),
#     ('precision', 'Precision'),
#     ('recall', 'Recall')
#     ('f1', 'F1'),
#     ('fbeta', 'FBeta'),
#     ('roc', 'ROC'),
#     ('precision_recall_curve', 'PrecisionRecallCurve'),
#     ('mAP','MeanAveragePrecision'),
#     ('fid','FID'),
#     ('cosine_similarity', 'CosineSimilarity'),
#     ('BLEUScore', 'BLEUScore'),
#     ('SQuAD', 'SQuAD'),
#     ('WER','WER'),
# ]
# )

class Experiment(pl.LightningModule):

    def __init__(self,
                 model: Base,
                 args: Namespace) -> None:
        super(Experiment, self).__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.args = args
        self.acc = Accuracy(threshold=args.threshold)
        self.auc = AUROC()
        self.pre = Precision(threshold=args.threshold)
        self.recall = Recall(threshold=args.threshold)
        self.prcurv = PrecisionRecallCurve()
        self.prauc = AUC()
    
    # def prepare_metrics(self):
    #     self.metircs = []
    #     metric_names = self.args.metrics_name.split(',')
    #     for m in metric_names:
    #         m = METRIC_MAPPER[m]
    #         mod = import_module('torchmetrics.'+m)
    #         self.metircs.append(mod())


    def compute_metrics(self, preds, target):
        result = preds
        if isinstance(preds, Tuple):
            result = preds[0]
        elif isinstance(preds, List):
            result = preds[0]
        elif isinstance(preds, Dict):
            result = preds['result']

        self.acc.to(result.device)
        self.auc.to(result.device)
        self.pre.to(result.device)
        self.recall.to(result.device)
        if result.dim() >= 2:
            pred_pr = F.softmax(result, dim=-1)
            self.acc.update(pred_pr[...,-1], target)
            self.auc.update(pred_pr[...,-1], target)
            self.pre.update(pred_pr[...,-1], target)
            self.recall.update(pred_pr[...,-1], target)
        elif result.dim() == 1:
            self.acc.update(result, target)
            self.auc.update(result, target)
            self.pre.update(result, target)
            self.recall.update(result, target)
        
    def log_metrics(self, prefix):
        if self.args.label is not None:
            acc_val = self.acc.compute()
            auc_val = self.auc.compute()
            precision_val = self.pre.compute()
            recall_val = self.recall.compute()

            self.log(name=prefix + "_acc", value=acc_val, sync_dist=True)
            self.log(name=prefix + "_auc", value=auc_val, sync_dist=True)
            self.log(name=prefix + "_precision", value=precision_val, sync_dist=True)
            self.log(name=prefix + "_recall", value=recall_val, sync_dist=True)   

    def reset_metrics(self):
        self.acc.reset()
        self.auc.reset()
        self.pre.reset()
        self.recall.reset()
        self.prcurv.reset()
        self.prauc.reset()


    def forward(self, **kwargs) -> torch.Tensor:
        return self.model(kwargs)

    
    def configure_example_input(self, data):
        self._example_input_array = self.prepare_inputs(data)
    
    def prepare_inputs(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Dict):
            return type(data)({k: self.prepare_inputs(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self.prepare_inputs(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.as_tensor(data).to(self.device)
        else:
            return data

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        inputs =self.prepare_inputs(batch)
        if self._example_input_array is None:
            self._example_input_array = inputs
        train_loss = self.model.loss(inputs)
        self.log(name='train_loss', value=train_loss, sync_dist=True)
        if self.args.label is not None:
            result = self.model(inputs)
            self.compute_metrics(result, batch[self.args.label])
            self.log_metrics('train')
            self.reset_metrics()
        return train_loss

    def training_epoch_end(self, training_step_outputs):
        self.reset_metrics()
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        inputs =self.prepare_inputs(batch)
        val_loss = self.model.loss(inputs)
        self.log(name='val_loss', value=val_loss, sync_dist=True)

        if self.args.label is not None:
            result = self.model(inputs)
            self.compute_metrics(result, batch[self.args.label])
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log_metrics(prefix='validation')
        self.reset_metrics()
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.lr,
                               weight_decay=self.args.weight_decay)

        

        return {'optimizer': optimizer}

    def lr_schedulers(self):
        optimizer = self.configure_optimizers()['optimizer']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = self.args.scheduler_gamma)
        return {'scheduler': scheduler}


class AutoDateSet(pl.LightningDataModule):
    def __init__(
        self,
        dataset: List[Dataset],
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = False,
        collate_fn = None,
        **kwargs,
    ):
        super(AutoDateSet, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = dataset  
        self.collate_fn = collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        if len(self.dataset) == 1 :
            self.train_dataset = self.dataset[0]
            self.val_dataset = self.dataset[0]
            self.test_dataset = self.dataset[0]
        elif len(self.dataset) == 2:
            self.train_dataset = self.dataset[0]
            self.val_dataset = self.dataset[1]
            self.test_dataset = self.dataset[1]
        elif len(self.dataset) == 3:
            self.train_dataset = self.dataset[0]
            self.val_dataset = self.dataset[1]
            self.test_dataset = self.dataset[2]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=True,
        )
        
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            drop_last=True,
        )


def train(args, experiment:Experiment, data: AutoDateSet):
    args.gpus = [i for i in range(torch.cuda.device_count())]
    logger =  TensorBoardLogger(save_dir=args.log_dir, log_graph=True, name=args.name, version=args.version)
    runner = pl.Trainer(logger=logger, 
                         callbacks=[
                            LearningRateMonitor(),
                            ModelCheckpoint(save_top_k=2, 
                                            dirpath =os.path.join(args.save_dir, "checkpoints"), 
                                            monitor= args.monitor,
                                            save_last= True),
                            EarlyStopping(monitor=args.monitor),
                            TQDMProgressBar()],
                         plugins=DDPPlugin(find_unused_parameters=args.find_unused_parameters) if len(args.gpus)>1  else None,
                         profiler=PyTorchProfiler(filename='profile'),
                         gpus=args.gpus,
                         strategy='deepspeed' if torch.cuda.device_count() > 0 else None,
                         max_epochs=args.num_epochs)
    runner.fit(model=experiment, datamodule=data)

    return runner
    

def get_train_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('train_base_args')
    group.add_argument("--batch_size", type=int, default=8)
    group.add_argument("--lr", type=float, default=0.001)
    group.add_argument("--weight_decay", type=float, default=0.0)
    group.add_argument("--use_kl_loss", action="store_true")
    group.add_argument("--use_cl_loss", action="store_true")
    group.add_argument("--save_dir", type=str,  default='save')
    group.add_argument("--log_dir", type=str,  default='logs')
    group.add_argument("--name", type=str,  default='tensorboard')
    group.add_argument("--version", type=str,  default='v1')
    group.add_argument("--monitor", type=str, default='val_loss')
    group.add_argument("--scheduler_gamma", type=float, default=0.99)
    group.add_argument("--num_epochs", type=int, default=100)
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--pin_memory", action="store_true")
    group.add_argument("--gpus", type=int, nargs='+', default=None)
    group.add_argument("--find_unused_parameters", action="store_true")
    group.add_argument("--label", type=str, default=None)
    group.add_argument("--threshold", type=float, default=0.5)
    group.add_argument("--temperature", type=float, default=0.9)
    group.add_argument("--debiased", action="store_true")
    group.add_argument("--tau_plus", type=float, default=0.1)
    args, unknow = parser.parse_known_args()
    return args




 