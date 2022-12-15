import inspect
import json
import math
import os
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from nltk import word_tokenize
import numpy as np

from packaging import version

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
from transformers.utils import logging
from PreTrainedModel import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    EvaluationStrategy,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_numpify,
    nested_xla_mesh_reduce,
    set_seed,
)
from tensorboardX import SummaryWriter
_use_native_amp = False
_use_apex = False
EPS = 1e-12
INIT_GUMBEL_TEMP = 5.0

control_lst = ['positive', 'negative', 'neutral']
Control_Temp = {'positive': 3967, 'negative':4633, 'neutral':8500}
control_Map = [jt.Var([3967]).long(), jt.Var([4633]).long(), jt.Var([8500]).long()]
sst_lst = [(0, 2), (1, 3), (4,)]
sst_standard = ["positive", "negative", "very positive", "very negative", "neutral"]
# Control_?Map = {j:i for i, j in enumerate(control_lst)}

logger = logging.get_logger(__name__)


class Trainer_Prefix:
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        model_gpt2: Optional[PreTrainedModel] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[jt.optim.Optimizer, jt.optim.LambdaLR] = (None, None),
        task_mode: Optional[str] = None,
        use_dropout: Optional[bool] = False,
        distill: Optional[bool] = False,
        matching_objective:Optional[str]= None,
        finetuned_gpt2: Optional[PreTrainedModel] = None,
        **kwargs,
    ):
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args
        set_seed(self.args.seed)
        # Seed must be set before instantiating the model when using model
        assert (
            model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` argument."
        assert model_init is None
        self.model = model
        self.gpt2 = model_gpt2
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.task_mode = task_mode
        self.use_dropout = use_dropout

        self.curr_best_eval = 10000000.

        self.distill = distill
        if self.distill:
            self.matching_objective = matching_objective
            self.finetuned_gpt2 = finetuned_gpt2
        
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        self.tb_writer = tb_writer
        self.log_history = []
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a future version. Use `args.prediction_loss_only` instead.",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        if tb_writer is None and is_tensorboard_available():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        
        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False
        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                    "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                    + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )


