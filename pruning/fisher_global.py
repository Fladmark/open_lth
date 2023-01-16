# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np
import datasets

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask
import torch.nn.functional as F
import torch



@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.975
    pruning_layers_to_ignore: str = None
    dataset_name = "mnist"

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        batch_size = 1
        dataset_hparams = hparams.DatasetHparams(pruning_hparams.dataset_name, batch_size)
        dataloader = datasets.registry.get(dataset_hparams)

        val_dict = {k: v for k, v in trained_model.named_parameters()}
        fisher_dict = {}
        grad_dict = {}
        for key in val_dict.keys():
            grad_dict[key] = torch.zeros(val_dict[key].shape)
            fisher_dict[key] = torch.zeros(val_dict[key].shape)

        num_items = 10000/batch_size

        for idx, (data, target) in enumerate(dataloader):
            output = trained_model(data)
            temp_loss = F.nll_loss(output, target)
            temp_loss.retain_grad()
            temp_loss.backward()
            for key in fisher_dict.keys():
                grad_dict[key] += {k:v.grad**2 for k,v in trained_model.named_parameters()}[key]
            trained_model.loss_criterion.zero_grad()
            if idx == num_items:
                break

        for key in val_dict.keys():
            fisher_dict[key] = 1 / (2 * num_items) * val_dict[key] ** 2 * grad_dict[key]

        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in fisher_dict.items()
                   if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]
        print(threshold)

        new_mask = Mask({k: np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))
                         for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
