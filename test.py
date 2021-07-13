from copy import deepcopy
import pathlib
from typing import Any, Sequence, Iterable, Dict, Tuple, Mapping
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import gc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict
import random
import os
import shutil
from fewshot import make_challenge
import random
import numpy as np
import torch
from fewshot import Model

from unifew.model import Unifew, UnifewDataset


TMP_CKPT_DIR = "/dev/shm"  # directory to save checkpoints during metatest (for comparing with lm_bff only)


def _get_filename(start, stop):
    if not start and not stop:
        return "predictions.json"
    else:
        return f"predictions_{start}-{stop}"


class MetatestDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PTLWrapperModel(Model):
    def __init__(
        self,
        seed: int,
        model_type: str,
        k=1,
        question=None,
        ckpt_path=None,
        output_path=None,
        **kwargs,
    ):
        super().__init__()

        args = argparse.Namespace(
            seed=seed,
            model_type=model_type,
            question=question,
            batch_size=1,
            k=k,
            predictions_path=output_path,
            **kwargs,
        )
        # self.model = Unifew(args, model, tokenizer, max_len, question)
        if ckpt_path is not None:
            self.model = Unifew.load_from_checkpoint(ckpt_path)
        else:
            self.model = Unifew(args.default_model)
        if kwargs.get("trainer"):
            self.model.args.trainer.max_steps = kwargs.get("trainer").get("max_steps")
            args.max_steps = kwargs.get("trainer").get("max_steps")
        if kwargs.get("lr"):
            self.model.args.lr = kwargs.get("lr")
        if kwargs.get("use_constant_schedule"):
            setattr(self.model.args, "use_constant_schedule", True)
        self.tokenizer = self.model.tokenizer
        self.max_len = self.model.max_len  # prediction length
        self.k = k
        assert question is not None
        self.question_str = question
        self.args = args
        self.model.train_dataloader = self.model.val_dataloader = self.model.test_dataloader = None
        self.model.args.predictions_path = self.args.predictions_path
        self.model_orig_state_dict = deepcopy(self.model.state_dict())
        # trainer.fit ignores the passed dataloaders if model already has loaders
        # we have to remove the existing dataloaders so the pl.trainer uses the passed data_loader
        try:
            if hasattr(Unifew, "train_dataloader"):
                delattr(Unifew, "train_dataloader")
            if hasattr(Unifew, "val_dataloader"):
                delattr(Unifew, "val_dataloader")
            if hasattr(Unifew, "test_dataloader"):
                delattr(Unifew, "test_dataloader")
        except AttributeError:
            pass

    def fit_and_predict(
        self,
        support_x: Iterable[Mapping[str, Any]],
        support_y: Iterable[str],
        target_x: Iterable[Mapping[str, Any]],
        metadata: Dict[str, Any] = None,
        val_x=None,
        val_y=None,
    ) -> Tuple[Sequence[str], Sequence[float]]:

        # filter out empty support examples
        valid_indices = [idx for idx in range(len(support_x)) if support_x[idx] and support_x[idx]["txt"]]
        support_x = [support_x[idx]["txt"] for idx in valid_indices]
        support_y = [support_y[idx] for idx in valid_indices]

        # here labels are passed in as string, convert to int
        support_y = list(map(int, support_y))
        target_x = [el["txt"] for el in target_x]
        # setting for LM-BFF, use half examples as val.

        def split_for_lm_bff(support_x, support_y):
            """utility function to split the 32 shots into 16 per each label
            This is only for comparison with lm_bff"""
            label_mappings = defaultdict(list)
            for i, el in enumerate(support_y[0]):
                label_mappings[el].append(i)
            # this is only for lm_bff baseline comparison, shots are 32 in that dataset
            assert (len(label_mappings[0])) == 32
            train_x, train_y = [], []
            val_x, val_y = [], []
            for lbl, index_list in label_mappings.items():
                for idx in index_list[:16]:
                    train_x.append(support_x[0][idx])
                    train_y.append(support_y[0][idx])
                for idx in index_list[16:]:
                    val_x.append(support_x[0][idx])
                    val_y.append(support_y[0][idx])

            def two_list_shuffle(list1, list2):
                """shuffle two lists in the same order"""
                temp = list(zip(list1, list2))
                random.shuffle(temp)
                return zip(*temp)

            train_x, train_y = two_list_shuffle(train_x, train_y)
            val_x, val_y = two_list_shuffle(val_x, val_y)
            return train_x, train_y, val_x, val_y

        if val_x is not None:
            valid_indices = [idx for idx in range(len(val_x)) if val_x[idx] and val_x[idx]["txt"]]
            val_x = [val_x[idx]["txt"] for idx in valid_indices]
            val_y = [val_y[idx] for idx in valid_indices]
            # here labels are passed in as string, convert to int
            val_y = list(map(int, val_y))
            # metadata, tokenizer, maxlen, question, subset, args
            val_ds = list(
                UnifewDataset.get_current_set_separate_train_test(
                    val_x,
                    target_x,
                    val_y,
                    None,
                    metadata,
                    self.tokenizer,
                    self.max_len,
                    self.question_str,
                    subset="train",
                    args=self.args,
                )
            )
        elif getattr(self.args, "lmbff_setting", False):  # only for comparisson with lm_bff
            support_x, support_y, val_x, val_y = split_for_lm_bff(support_x, support_y)
            # subset should be 'train' because of the way this function is designed
            val_ds = list(
                UnifewDataset.get_current_set_separate_train_test(
                    val_x,
                    target_x,
                    val_y,
                    None,
                    metadata,
                    self.tokenizer,
                    self.max_len,
                    self.question_str,
                    subset="train",
                    args=self.args,
                )
            )
        else:
            val_ds = []
        train_ds = list(
            UnifewDataset.get_current_set_separate_train_test(
                support_x,
                target_x,
                support_y,
                None,
                metadata,
                self.tokenizer,
                self.max_len,
                self.question_str,
                subset="train",
                args=self.args,
            )
        )
        test_ds = list(
            UnifewDataset.get_current_set_separate_train_test(
                support_x,
                target_x,
                support_y,
                None,
                metadata,
                self.tokenizer,
                self.max_len,
                self.question_str,
                subset="val",
                args=self.args,
            )
        )

        # data is already batched, so set batch_size=1
        train_loader = DataLoader(MetatestDataset(train_ds), batch_size=1, collate_fn=UnifewDataset.collate_fn)
        val_loader = (
            DataLoader(
                MetatestDataset(val_ds),
                batch_size=1,
                collate_fn=UnifewDataset.collate_fn,
            )
            if val_ds
            else None
        )
        test_loader = DataLoader(MetatestDataset(test_ds), batch_size=1, collate_fn=UnifewDataset.collate_fn)
        self.args.num_sanity_val_steps = 0
        self.args.gpus = 1

        for p in self.model.parameters():
            p.requires_grad = True
        dirpath = ""

        # train the model using the support examples in the episode
        if self.k > 0:  # few shot
            if all([el[0].nelement() != 0 for el in train_ds]):  # check if support exists
                if len(val_ds) > 0:
                    dirpath = f"{TMP_CKPT_DIR}/unifew-{self.args.start}-{self.args.stop}/"
                    pathlib.Path(dirpath).parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_callback = ModelCheckpoint(
                        monitor="avg_val_acc",
                        mode="max",
                        save_top_k=1,
                        filepath=dirpath + "unifew-{avg_val_acc:.3f}",
                    )
                    trainer = pl.Trainer.from_argparse_args(self.args, checkpoint_callback=checkpoint_callback, logger=None)
                    trainer.fit(self.model, train_loader, val_loader)
                else:
                    trainer = pl.Trainer.from_argparse_args(self.args, checkpoint_callback=False, logger=None)
                    trainer.fit(self.model, train_loader)
            else:
                trainer = pl.Trainer.from_argparse_args(self.args, checkpoint_callback=False, logger=None)
                print("skipping training for this episode, no support examples")
        else:
            trainer = pl.Trainer.from_argparse_args(self.args, checkpoint_callback=False, logger=None)

        # test on query examples
        for p in self.model.parameters():
            p.requires_grad = False
        if len(val_ds) > 0:  # load model from checkpoint
            results = trainer.predict(dataloaders=test_loader)
        else:
            results = trainer.predict(self.model, dataloaders=test_loader)

        predictions = torch.cat(results).tolist()
        # dummy scores, the model doesn't support scoring predictions atm
        scores = [1.0 for _ in predictions]

        # we restore the original state dict of the model before running the next episode
        state_dict_to_load = deepcopy(self.model_orig_state_dict)
        self.model.load_state_dict(state_dict_to_load)
        torch.cuda.empty_cache()
        if dirpath and os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        del trainer
        gc.collect()
        return predictions, scores


@hydra.main(config_path="conf", config_name="test")
def main(cfg: DictConfig) -> None:
    # we pass in start and stop for checkpointing
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    model = instantiate(cfg.model, start=cfg.start, stop=cfg.stop, default_model=cfg.default_model)
    evaluator = make_challenge(cfg.challenge)
    evaluator.save_model_predictions(
        model=model,
        save_path=_get_filename(cfg.start, cfg.stop),
        start_task_index=cfg.start,
        stop_task_index=cfg.stop,
    )


if __name__ == "__main__":
    main()
