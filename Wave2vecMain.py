from typing import Dict, List, Optional, Union

import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

from model.wave2vec import WrapWave2vec
from experiment import Experiment, AutoDateSet, train, get_args


class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(
            batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self.model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1)).to(
                torch.long
            )

            attention_mask = torch.zeros(
                (batch_size,
                 mask_indices_seq_length), dtype=torch.long, device=batch["input_values"].device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            attention_mask[(torch.arange(attention_mask.shape[0], 
                                         device=batch["input_values"].device), 
                                         output_lengths - 1)] = 1
                                         
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=attention_mask,
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(
            mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(
            sampled_negative_indices, dtype=torch.long, device=device)

        return batch


def get_model(ars):
    return WrapWave2vec(ars)


def get_datasets(args):
    return None


def main():
    args = get_args()

    model = get_model(args)
    datasets = get_datasets(args)

    data_collator = DataCollatorForWav2Vec2Pretraining(model=model.model, 
                                                       feature_extractor=model.feature_extractor, 
                                                       pad_to_multiple_of=args.pad_to_multiple_of
                                                       )

    experiment = Experiment(model=model, args=args)
    data = AutoDateSet(datasets, args.batch_size, args.batch_size,
                       args.num_workers, args.pin_memory, collate_fn=data_collator)
    train(args, experiment, data)


if __name__ == "__main__":
    main()
