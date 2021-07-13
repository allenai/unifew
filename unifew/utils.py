import logging
import random
import re

import fewshot
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def assemble_sampler_cfg(cfg: DictConfig, split: str) -> DictConfig:
    # Get challenge stores
    challenge_spec = fewshot.get_challenge_spec(cfg.challenge)
    if "val" in split:
        store_cfgs = challenge_spec.val_stores
    elif "train" in split:
        store_cfgs = challenge_spec.train_stores
    elif "test" in split:
        raise RuntimeError("Peeking at some test datasets!!!!")
    else:
        raise ValueError(f"Unrecognized split {split}")

    # Create datasets based on those stores, with the specified samplers
    sampler_cfg = instantiate(cfg.metadatasetsampler_cfg)
    sampler_cfg.datasets = [
        fewshot.datasets.DatasetCfg(
            labeled_store=store,
            sampler=instantiate(cfg.sampler),
        )
        for store in store_cfgs.values()
    ]
    return sampler_cfg


def create_batches(iterable, batch_size=None, tensorize=True, pad=False, padding_value=0.0):
    if batch_size is None:
        yield [e for e in iterable]
    else:
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            sequence = list(iterable[ndx : min(ndx + batch_size, l)])
            if tensorize:
                sequence = [torch.tensor(el) for el in sequence]
                if pad:
                    res = pad_sequence(sequence, padding_value=padding_value, batch_first=True)
                else:
                    assert len(set([el.shape[0] for el in sequence])) == 1
                    res = sequence
            else:
                res = sequence
            yield res


def normalize_label(label, valid_labels):
    """normalize a label (remove "(B) " from a prediction string like "(B) No". )"""
    res = re.sub(r"\([A-H]\) ", "", label)
    final_res = res
    if res not in valid_labels:
        for lbl in valid_labels:
            if lbl in res:
                final_res = lbl
                break
    if final_res not in valid_labels:
        # sometimes capitalization matters
        for lbl in valid_labels:
            if lbl.lower() == final_res.lower():
                final_res = lbl
                break
        else:
            final_res = random.choice(valid_labels)
    return final_res
