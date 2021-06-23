import logging
from omegaconf import DictConfig
import fewshot
from hydra.utils import instantiate

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
