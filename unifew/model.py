import logging
import string
from typing import Callable, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import re
from sklearn.metrics import accuracy_score, f1_score

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset
from transformers import Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from unifew.utils import assemble_sampler_cfg, create_batches, normalize_label

logger = logging.getLogger(__name__)


class UnifewDataset(IterableDataset):
    def __init__(self, args, tokenizer=None, question=None, max_len=None, subset="train"):
        worker_info = torch.utils.data.get_worker_info()
        random_seed = 0 if worker_info is None else int(worker_info.id)
        sampler_cfg = assemble_sampler_cfg(args, split=subset)
        sampler_cfg.seed = random_seed
        # assert sampler_cfg.dataloader_batch_size == 1
        self.metadatasampler = hydra.utils.instantiate(sampler_cfg)
        self.tokenizer = tokenizer
        UnifewDataset.tokenizer = tokenizer
        self.args = args
        self.subset = subset
        self.question = question
        self.max_len = max_len

    @staticmethod
    def encode_question_and_label(
        text,
        question_str,
        tokenizer=None,
        maxlen=1024,
        all_labels=None,
        task_type="generic",
        args=None,
    ):
        """encode text and label in a single context
        Args:
            text: the text we want to classify
            label: the label
            is_predict: is the label available?
            question_str: default question used to convert the classification sample into qa sample
            truncate: if True, it truncates the text+question+label to self.args.max_len
            tokenizer: tokenizer to use
            all_labels: all possible textual labels for the task
            task_type: type of the task
        """
        # original unifiedqa formatting:
        # fout.write(f"{question} \\n (A) {answerA} (B) {answerB} (C) {answerC} \\n {context} \t {ans}\n")
        def is_nli(s):
            # return True if 'scitail' in metadata[0]['dataset'] or 'snli' in metadata[0]['dataset'] else False
            return True if "###" in text and s.count("###") == 1 else False

        def is_fewrel(text):
            return True if text.count("*") == 2 and text.count("#") == 2 else False

        text = text.replace("\n", " ").replace("\t", " ")

        if is_nli(text):
            final_text = ""
        elif is_fewrel(text):  # entity relationship
            try:
                mention1 = re.search(r"# ([\w\W]+) #", text).group(1)
                mention2 = re.search(r"\* ([\W\w]+)? \*", text).group(1)
                final_text = f"{mention1} to {mention2}? \\n"
            except AttributeError:
                # roll back to the default question (in case of very rare weird formatting)
                final_text = f"{question_str} \\n"
        elif getattr(args, "use_subj_format", False) and task_type == "subj":
            final_text = f"The language of this statement is? \\n"
        elif getattr(args, "use_trec_format", False) and task_type == "trec":
            final_text = f"The question type? \\n"
        elif getattr(args, "use_conll_format", True) and task_type == "conll":
            final_text = f"What is the type of the entity between the # marks?"
        else:
            final_text = f"{question_str} \\n"

        # add label strings and answers
        for i, e in enumerate(all_labels):
            final_text += f" ({string.ascii_uppercase[i]}) {e} "
        if not is_nli(text):
            final_text += f" \\n {text}"
        else:
            premise, hypothesis = text.split("###")
            # for NLI we use a slightly different formatting, following Gao et al (2020)
            final_text = f"{premise} Is {hypothesis}? \\n"
            for i, e in enumerate(all_labels):
                final_text += f" ({string.ascii_uppercase[i]}) {e} "
        context_tokens = tokenizer.encode(final_text, add_special_tokens=True, padding="longest", truncation=True)

        return context_tokens

    @staticmethod
    def get_current_set_separate_train_test(
        support_x,
        query_x,
        support_y,
        query_y,
        metadata,
        tokenizer,
        maxlen,
        question,
        subset,
        args,
    ):
        "breaks the support and target and returns"
        # assert len(query_x) == len(support_x) == len(support_y) == 1
        metadata, task_type = UnifewDataset.format_labels(metadata, query_x)

        idx_to_label = {v: k for k, v in metadata["mapped_labels"].items()}

        def encode_and_tokenize_docs(doc_list):
            """
            for a list of documents and corresponding labels, convert each to one single
            QA instance and tokenize it
            """
            return [
                UnifewDataset.encode_question_and_label(
                    text=example,
                    question_str=question,
                    tokenizer=tokenizer,
                    maxlen=maxlen,
                    all_labels=list(metadata["mapped_labels"].keys()),
                    task_type=task_type,
                    args=args,
                )
                for example in doc_list
            ]

        if query_y is not None:
            # this is for meta-train time
            # in unifiedqa we use both support and query at training time to update model parameters
            docs = query_x + support_x
            labels = np.concatenate((query_y, support_y))

            labels = [idx_to_label[lbl] for lbl in labels]

            textual_label_token_ids = tokenizer.batch_encode_plus(labels, padding=True)["input_ids"]
            # list of lists of token ids
            doc_token_ids_list = encode_and_tokenize_docs(docs)
            label_token_ids_batches = [
                torch.tensor(e)
                for e in create_batches(
                    textual_label_token_ids,
                    args.query_batch_size,
                    tensorize=True,
                    pad=True,
                    padding_value=tokenizer.pad_token_id,
                )
            ]
        else:
            # meta-test time
            # using test.py this function is called twice for meta-test
            # (once with subset == 'train' and once with subset == 'dev'/'test')
            # once to get training examples from support set, and once to get test examples from the query set
            if subset != "train":
                # at test time, this is just for returning query set, ignores the support set
                doc_token_ids_list = encode_and_tokenize_docs(query_x)
                #  create batches of None
                label_token_ids_batches = [
                    None
                    for e in create_batches(
                        doc_token_ids_list,
                        args.query_batch_size,
                        tensorize=True,
                        pad=True,
                        padding_value=tokenizer.pad_token_id,
                    )
                ]
            else:
                # when doing fine-tuning at meta-test time, we only use support examples and ignore query set
                query_x = support_x
                query_y = support_y
                query_labels = [idx_to_label[lbl] for lbl in query_y]

                if query_labels:
                    textual_label_token_ids = tokenizer.batch_encode_plus(
                        query_labels, add_special_tokens=True, padding=True
                    )["input_ids"]
                else:  # sometimes the episode does not provide any support examples
                    textual_label_token_ids = [[]]
                if query_x:
                    doc_token_ids_list = encode_and_tokenize_docs(query_x)
                else:  # sometimes there is no support example in the episode
                    doc_token_ids_list = [[]]
                label_token_ids_batches = [
                    torch.tensor(e)
                    for e in create_batches(
                        textual_label_token_ids,
                        args.query_batch_size,
                        tensorize=True,
                        pad=True,
                        padding_value=tokenizer.pad_token_id,
                    )
                ]
        doc_token_ids_batches = [
            torch.tensor(e)
            for e in create_batches(
                doc_token_ids_list,
                args.query_batch_size,
                tensorize=True,
                pad=True,
                padding_value=tokenizer.pad_token_id,
            )
        ]
        # iterate through all possible support and query batches
        for j in range(len(doc_token_ids_batches)):
            doc_token_ids_, label_token_ids_ = (
                doc_token_ids_batches[j],
                label_token_ids_batches[j],
            )
            yield doc_token_ids_, label_token_ids_, metadata

    def __iter__(self):
        return self.get_samples()

    @staticmethod
    def format_labels(metadata, query_x):
        """format labels"""
        nli_datasets = ["scitail", "nli", "rte"]
        # TODO: dataset information was not avaialbe in metadata in earlier versions of flex
        # We used to infer it from labels
        # some code here related to this is redundent and needs cleaning up
        sentiment_label_mapping = {"positive": "positive", "negative": "negative"}
        snli_label_mapping = {
            "neutral": "Maybe",
            "entailment": "Yes",
            "contradiction": "No",
            "not_entailment": "No",
        }
        subj_label_mapping = {"objective": "objective", "subjective": "subjective"}
        trec_label_mapping = {
            "description": "description",
            "entity": "entity",
            "number": "number",
            "location": "location",
            "expression": "expression",
            "human": "human",
        }

        current_lable_set = set(metadata["text_labels"].keys())

        mappings = None
        if not isinstance(metadata["dataset"], str):
            metadata["dataset"] = metadata["dataset"].name
        task_type = "generic"
        for dsname in nli_datasets:
            if dsname in metadata["dataset"]:
                mappings = snli_label_mapping
                task_type = "nli"
                break
        if current_lable_set == set(sentiment_label_mapping.keys()):
            mappings = sentiment_label_mapping
            task_type = "sentiment"
        elif current_lable_set == set(subj_label_mapping.keys()):
            task_type = "subj"
        elif current_lable_set == set(trec_label_mapping.keys()):
            task_type = "trec"
        if metadata["dataset"] == "conll":
            task_type = "conll"

        if mappings is not None:
            metadata["mapped_labels"] = {mappings[e]: v for e, v in metadata["text_labels"].items()}
        else:
            metadata["mapped_labels"] = metadata["text_labels"]
        metadata["mapped_labels"] = {k.replace("_", " "): v for k, v in metadata["mapped_labels"].items()}
        return metadata, task_type

    def get_samples(self):
        """get samples from each training/testing episode"""
        for epiosode_data in self.metadatasampler:
            support_x, query_x, support_y, query_y, metadata = epiosode_data
            # get batched examples from this episode
            self.current_set_gen = UnifewDataset.get_current_set_separate_train_test(
                support_x,
                query_x,
                support_y,
                query_y,
                metadata,
                tokenizer=self.tokenizer,
                maxlen=self.max_len,
                question=self.question,
                subset=self.subset,
                args=self.args,
            )
            for record in self.current_set_gen:
                yield record

    @staticmethod
    def collate_fn(batch):
        list_batch = list(zip(*batch))
        input_ids, label, metadata = list_batch
        return input_ids[0], label[0], metadata[0]


class Unifew(pl.LightningModule):
    def __init__(self, hparams, model=None, tokenizer=None, max_len=None) -> None:
        super(Unifew, self).__init__()
        self.hparams.update(hparams)
        self.args = hparams
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        if model is None:
            # instantiate model and tokenizer
            assert hparams.model_type == "unifew"
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-t5-large")
            max_len = model.config.n_positions

        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.question = hparams.question
        self.all_labels = {}

    def configure_optimizers(self):
        if self.args.model_type == "unifew" and getattr(self.args, "do_train", False):
            if getattr(self.args, "optimizer", "adam") == "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
                num_steps = self.args.trainer.max_steps
                warmup_steps = self.args.warmup or int(0.1 * num_steps)  # NOTE: Hardcoded warmup steps
                if getattr(self.args, "use_constant_schedule", False):
                    scheduler = get_constant_schedule(optimizer)
                else:
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=num_steps,
                    )
            elif self.args.optimizer == "adafactor":
                # recommended hyperparams from https://huggingface.co/transformers/main_classes/optimizer_schedules.html
                optimizer = Adafactor(
                    self.parameters(),
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                    lr=1e-3,
                    clip_threshold=1.0,
                )
                # recommended to use constant schedule with warmup https://huggingface.co/transformers/main_classes/optimizer_schedules.html
                scheduler = get_constant_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(0.1 * self.args.trainer.max_steps)
                )
            self.scheduler = scheduler
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_nb):
        support_token_ids, support_y_ids, metadata = batch
        if self.args.model_type == "unifew" and getattr(self.args, "do_train", False):
            output = self.model(input_ids=support_token_ids, labels=support_y_ids, return_dict=True)
            loss = output.loss
            learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
            tensorboard_logs = {
                "lr": learning_rate,
                "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0,
                "tr_loss": loss.detach().item(),
            }
            for k, v in tensorboard_logs.items():
                self.log(k, v)
            return loss
        else:
            # return an unused loss value, otherwise PL throws errors
            return support_token_ids.new_ones(1) * 1.0

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        if self.args.model_type == "unifew" and getattr(self.args, "do_train", False):
            super().backward(closure_loss, optimizer, opt_idx, *args, **kwargs)
        else:
            pass  # no grad update

    def _validation_step(self, batch, batch_nb, is_test=False):
        for p in self.model.parameters():
            p.requires_grad = False

        query_token_ids, query_y_ids, metadata = batch

        label_to_idx = metadata["mapped_labels"]

        for key in metadata["mapped_labels"]:
            if key not in self.all_labels:
                self.all_labels[key] = len(self.all_labels)

        generated_sequence = self.model.generate(input_ids=query_token_ids)
        texts = [
            self.tokenizer.decode(seq, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            for seq in generated_sequence
        ]
        if query_y_ids is not None:  # calculate loss
            output = self.model(
                input_ids=query_token_ids,
                labels=query_y_ids,
                use_cache=False,
                return_dict=True,
            )
            loss = output.loss
            label_str_orig = [self.tokenizer.decode(lbl, skip_special_tokens=True) for lbl in query_y_ids]
            labels = [label_to_idx[lbl] for lbl in label_str_orig]
            global_label_orig = [self.all_labels[e] for e in label_str_orig]
            global_label_orig = torch.tensor(global_label_orig, device=query_token_ids.device, dtype=torch.int)
        else:
            global_label_orig = None
            labels = None
            loss = query_token_ids.new_ones(1) * 0.01
        pred_labels = []
        global_pred_labels = []

        for predicted_label_str in texts:
            if predicted_label_str not in label_to_idx:
                predicted_label_str = normalize_label(predicted_label_str, list(label_to_idx.keys()))
            predicted_label = label_to_idx[predicted_label_str]
            pred_labels.append(predicted_label)
            global_label_pred = self.all_labels[predicted_label_str]
            global_pred_labels.append(global_label_pred)

        global_pred_labels = torch.tensor(global_pred_labels, device=query_token_ids.device, dtype=torch.int)
        pred_labels = torch.tensor(pred_labels, device=query_token_ids.device, dtype=torch.int)
        step = torch.tensor(self.trainer.global_step, device=query_token_ids.device, dtype=torch.int)
        return {
            "prediction": pred_labels,
            "label": labels,
            "global_label": global_pred_labels,
            "global_label_orig": global_label_orig,
            "val_loss": loss.detach().item(),
            "step": step,
        }

    def validation_step(self, batch, batch_nb):
        return self._validation_step(batch, batch_nb, is_test=False)

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True
        predictions = torch.cat([x["global_label"] for x in outputs])
        labels = torch.cat([x["global_label_orig"] for x in outputs])
        preds_1d = predictions.tolist()  # move to cpu
        labels_1d = labels.tolist()  # move to cpu
        f1 = f1_score(labels_1d, preds_1d, average="macro")
        acc = accuracy_score(labels_1d, preds_1d)
        logs = {"avg_val_acc": acc, "avg_val_f1": f1, "step": outputs[0]["step"]}
        for k, v in logs.items():
            self.log(k, v)
        return {"avg_val_acc": acc, "avg_val_f1": f1, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_nb):
        result = self._validation_step(batch, batch_nb, is_test=True)
        return {f"test_{k}": v for k, v in result.items()}

    def test_epoch_end(self, outputs):
        predictions = torch.cat([x["test_prediction"] for x in outputs])
        return {"predictions": predictions}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        res = self._validation_step(batch, batch_idx, is_test=False)
        return res["prediction"]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        if self.args.model_type == "unifew" and getattr(self.args, "do_train", False):
            super().optimizer_step(
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx,
                optimizer_closure,
                on_tpu,
                using_native_amp,
                using_lbfgs,
            )
            # optimizer.step()
            # optimizer.zero_grad()  # prob not needed as PL already does this through model.optimizer_zero_grad
        else:
            pass

    def _get_data_loader(self, subset):
        assert (
            self.args.batch_size == 1
        )  # batching is handeled within the dataset with `query_batch_size`, DataLoader shouldn't batch again

        logger.info(f"loading {subset} data ...")
        ds = UnifewDataset(
            args=self.args,
            tokenizer=self.tokenizer,
            question=self.question,
            max_len=self.max_len,
            subset=subset,
        )
        dl = DataLoader(
            ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=UnifewDataset.collate_fn,
        )
        logger.info(f"...loaded {subset} set")
        return ds, dl

    def train_dataloader(self):
        if self.train_dataloader_object is not None:
            return self.train_dataloader_object

        ds, dl = self._get_data_loader("train")

        self.train_dataloader_object = dl
        return self.train_dataloader_object

    def val_dataloader(self):
        if self.val_dataloader_object is not None:
            return self.val_dataloader_object

        ds, dl = self._get_data_loader("val")

        self.val_dataloader_object = dl
        return self.val_dataloader_object

    def test_dataloader(self):
        if self.test_dataloader_object is not None:
            return self.test_dataloader_object

        ds, dl = self._get_data_loader("test")

        self.test_dataloader_object = dl
        return self.test_dataloader_object
