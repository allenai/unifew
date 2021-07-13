import os
import random

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from unifew.model import Unifew


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # cfg = OmegaConf.to_container(cfg, resolve=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = Unifew(cfg)

    tensorboad_logger = TensorBoardLogger(save_dir=cfg.save_dir, name=cfg.save_prefix, version=0)  # always use version=0
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(cfg.save_dir, cfg.save_prefix),
        filename="{epoch}-{step}-{avg_val_acc:.2f}",
        monitor="avg_val_acc",
        mode="max",
        save_top_k=cfg.save_top_k,
    )

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        distributed_backend="ddp" if cfg.trainer.gpus > 1 else None,
        logger=tensorboad_logger,
        checkpoint_callback=model_ckpt,
    )
    if not cfg.test_only:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
