import json
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
import numpy as np


import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
from jittor.dataset.sampler import Sampler, SequentialSampler, RandomSampler
from transformers.utils import logging
from pretrainedmodel import PreTrainedModel
from tqdm.auto import tqdm, trange
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
from warmup import *
_use_native_amp = False
_use_apex = False
EPS = 1e-12
INIT_GUMBEL_TEMP = 5.0


def select_index(metrix: jt.Var, index: jt.Var, dim: int = 0) -> jt.Var:
    return metrix[(slice(None),)*dim+(index,)]

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

        # if is_datasets_available():
        #     if isinstance(train_dataset, datasets.Dataset):
        #         self._remove_unused_columns(self.train_dataset, description="training")
        #     if isinstance(eval_dataset, datasets.Dataset):
        #         self._remove_unused_columns(self.eval_dataset, description="evaluation")

        self.global_step = None
        self.epoch = None
        self.total_flos = None
        # if self.args.fp16 and _use_native_amp:
        #     self.scaler = torch.cuda.amp.GradScaler()
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        if self.args.label_names is None:
            self.args.label_names = (
                ["start_positions, end_positions"]
                if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
                else ["labels"]
            )
            
    def _get_train_sampler(self) -> Optional[Sampler]:
        return RandomSampler(self.train_dataset)
    
    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[Sampler]:
        return SequentialSampler(eval_dataset)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
            ]


            self.optimizer = jt.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )


            # for n, p in self.model.named_parameters():
            #     print(n,p.requires_grad)
            print(self.optimizer.state_dict())
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def num_examples(self, dataset: Dataset) -> int:
        return len(dataset)
    

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        # self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        # ================ dataset loader ===============
        self.train_dataset.batch_size = self.args.train_batch_size
        self.train_dataset.sampler = self._get_train_sampler()
        self.train_dataset.collate_batch = self.data_collator
        self.train_dataset.drop_last = self.args.dataloader_drop_last
        self.train_dataset.num_workers = self.args.dataloader_num_workers
        # ================ end ===========================
        train_dataloader = self.train_dataset
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                jt.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(jt.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model


        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                # print(model, model.module)
                # if self.args.n_gpu > 1:
                #     self.total_flos = getattr(model.module.config, "total_flos", 0)
                # else:
                self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = jt.Var(0.0)
        logging_loss_scalar = 0.0
        # model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        # train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):

            epoch_iterator = train_dataloader
            ppl = None
            train_loss = None
            eval_loss = None

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None
            with tqdm(total = epoch_iterator.__batch_len__()) as bar:
            # epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
                for step, inputs in enumerate(epoch_iterator):

                    # print([self.tokenizer.decode(i) for i in inputs['input_ids'].data])
                    # labels = np.copy(inputs['labels'].data)
                    # labels[labels==-100] = 50257
                    # print([self.tokenizer.decode(i) for i in labels])
                    # print([self.tokenizer.decode(i) for i in inputs['src'].data])

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        # epoch_pbar.update(1)
                        continue

                    train_loss = self.training_step(model, inputs)
                    tr_loss += train_loss

                    self.total_flos += self.floating_point_ops(inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):

                        self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        # print([(name,p.requires_grad) for name, p in self.model.named_parameters()])
                        # print('=======================================================================')
                        # print([(name,p.requires_grad) for name, p in self.gpt2.named_parameters()])
                        self.optimizer.step()

                        # URGENT
                        self.lr_scheduler.step()
                        # model.zero_grad()
                        self.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)


                        if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                        ):
                            logs: Dict[str, float] = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            # backward compatibility for pytorch schedulers
                            # logs["learning_rate"] = (
                            #     self.lr_scheduler.get_last_lr()[0]
                            #     # if version.parse(jt.__version__) >= version.parse("1.4")
                            #     # else self.lr_scheduler.get_lr()[0]
                            # )
                            logging_loss_scalar = tr_loss_scalar
                            self.tb_writer.add_scalar('loss/train_loss', logs['loss'], self.global_step)

                            self.log(logs)

                        # print(self.args.evaluation_strategy == EvaluationStrategy.STEPS )
                        # print(self.global_step % self.args.eval_steps == 0)
                        # print()

                        if (
                            self.args.evaluation_strategy == EvaluationStrategy.STEPS
                            and self.global_step % self.args.eval_steps == 0
                        ):
                            metrics = self.evaluate()

                            eval_loss = metrics["eval_loss"]
                            ppl = math.exp(metrics["eval_loss"])
                            self.tb_writer.add_scalar('loss/eval_loss', metrics["eval_loss"], self.global_step)
                            self.tb_writer.add_scalar('ppl', ppl, self.global_step)
                            #############################EARLY STOPPING########################
                            if 'lowdata' in self.args.output_dir or 'earlystop' in self.args.output_dir:
                                self.save_based_on_eval = True
                            else:
                                self.save_based_on_eval = False
                            tqdm.write('if not see a line lowdata: below, then did not go into low data. ')
                            if self.save_based_on_eval and metrics["eval_loss"] < self.curr_best_eval:
                                tqdm.write('lowdata:', self.global_step, self.curr_best_eval, metrics["eval_loss"],
                                    'perplexity={}'.format(math.exp(metrics["eval_loss"])))
                                self.curr_best_eval = metrics["eval_loss"]

                                if hasattr(model, "module"):
                                    assert (
                                            model.module is self.model
                                    ), f"Module {model.module} should be a reference to self.model"
                                else:
                                    assert model is self.model, f"Model {model} should be a reference to self.model"
                                # Save model checkpoint
                                output_dir_name = os.path.basename(self.args.output_dir)
                                checkpoint_folder = f"{output_dir_name}-{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                                self.store_flos()
                                tqdm.write('saving to output_dir', output_dir)
                                self.save_model(output_dir)

                                self._rotate_checkpoints(use_mtime=True)
                            #####################################################

                        if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                            tqdm.write('saving model at a checkpoint!!')
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert (
                                    model.module is self.model
                                ), f"Module {model.module} should be a reference to self.model"
                            else:
                                assert model is self.model, f"Model {model} should be a reference to self.model"
                            # Save model checkpoint
                            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                            self.store_flos()

                            self.save_model(output_dir)
                            self._rotate_checkpoints(use_mtime=True)

                            jt.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            jt.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    bar.set_description('Epoch %i' % epoch)
                    bar.set_postfix(train_loss = train_loss.item(), eval_loss = eval_loss,ppl = ppl )
                    bar.update(1)
                    # epoch_pbar.update(1)
                    if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                        break
                # epoch_pbar.close()
                # train_pbar.update(1)

            if self.args.evaluation_strategy == EvaluationStrategy.EPOCH:
                metrics = self.evaluate()

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        # train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)


    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        # Set up loggers like W&B or Comet ML

        if hasattr(self, "_log"):
            warnings.warn(
                "The `_log` method is deprecated and won't be called in a future version, define `log` in your subclass.",
                FutureWarning,
            )
            return self._log(logs, iterator=iterator)

        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                logs["total_flos"] = self.total_flos
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        output = {**logs, **{"step": self.global_step}}
        # if self.is_world_process_zero():
        self.log_history.append(output)
        if iterator is not None:
            iterator.write(output)
        else:
            tqdm.write(str(output))

    def _prepare_inputs(self, inputs: Dict[str, Union[jt.Var, Any]]) -> Dict[str, Union[jt.Var, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, jt.Var):
                inputs[k] = v

        if self.args.past_index >= 0 and self._past is not None:
            assert  False
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[jt.Var, Any]]) -> jt.Var:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        if self.use_dropout:
            if self.gpt2 is not None:
                self.gpt2.train()
        inputs = self._prepare_inputs(inputs)


        if self.distill:
            loss = self.compute_loss_distill(model, inputs, gpt2_model=self.gpt2)
        else:
            loss = self.compute_loss(model, inputs, gpt2_model=self.gpt2)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps


        # print(loss)
        self.optimizer.backward(loss)

        # print('max allocated_memory:', torch.cuda.max_memory_allocated(0), 'total_memory:', torch.cuda.get_device_properties(0).total_memory,
        #       'percentage', torch.cuda.max_memory_allocated(0)/torch.cuda.get_device_properties(0).total_memory)


        return loss.detach()





    def compute_loss(self, model, inputs, gpt2_model=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # outputs = model.forward_weighted(**inputs)
        if 'prompt_lab' in inputs:
            prompt_lab_ = inputs['prompt_lab']
            k = jt.concat(self.discri_labels_code, dim=0)
            inputs['control_code'] = select_index(k, 0, prompt_lab_)
            del inputs['prompt_lab']

        outputs = model(**inputs, gpt2_model=gpt2_model)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # print(outputs[0])
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # print(outputs[0], outputs.loss)
        # URGENT
        # print('compute_loss', outputs[0])
        return outputs[0].mean()

    def compute_loss_distill(self, model, inputs, gpt2_model=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # outputs = model.forward_weighted(**inputs)

        with jt.no_grad():
            output_finetuned = self.finetuned_gpt2(**inputs)

        outputs = model(**inputs, gpt2_model=gpt2_model)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.matching_objective == 'kl':
            # distrib_finetuned=torch.log_softmax(output_finetuned.logits[:,:,:-2], dim=-1)  #bsz, seqlen, vocab
            distrib_finetuned=nn.log_softmax(output_finetuned.logits, dim=-1)  #bsz, seqlen, vocab
            distrib_prefix = nn.log_softmax(outputs.logits, dim=-1)  # bsz, seqlen, vocab
            loss = jt.sum(distrib_finetuned.exp() * (distrib_finetuned - distrib_prefix), dim=-1) #bsz, seqlen

        elif self.matching_objective == 'logits':
            loss = jt.norm(output_finetuned.logits - outputs.logits, dim=-1)  #bsz, seqlen
            # loss = torch.norm(output_finetuned.logits[:,:,:-2] - outputs.logits, dim=-1)  #bsz, seqlen

        elif self.matching_objective == 'last_layer':
            activation_diff = output_finetuned.last_hidden_state - outputs.last_hidden_state
            loss = jt.norm(activation_diff, dim=-1)  # bsz, seqlen
        else:
            assert False, "invalid matching_objective"

        return  loss.sum(dim=-1).mean()

    def is_local_master(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
        several machines) main process.

        .. warning::

            This method is deprecated, use :meth:`~transformers.Trainer.is_local_process_zero` instead.
        """
        warnings.warn("This method is deprecated, use `Trainer.is_local_process_zero()` instead.", FutureWarning)
        return self.is_local_process_zero()

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
        several machines) main process.
        """

        return self.args.local_rank in [-1, 0]



    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        self._save(output_dir)



    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        jt.save(self.args, os.path.join(output_dir, "training_args.bin"))
        json.dump(
            self.log_history, open(os.path.join(output_dir, "log_history.json"), "w"), indent=2, ensure_ascii=False
        )

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                self.model.config.total_flos = total_flos

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        output_dir_name = os.path.basename(self.args.output_dir)
        checkpoint_prefix = f"{output_dir_name}-{PREFIX_CHECKPOINT_DIR}"

        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        # ================= eval dataloader =================
        my_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        my_eval_dataset.batch_size = self.args.eval_batch_size
        my_eval_dataset.sampler = self._get_eval_sampler(eval_dataset=my_eval_dataset)
        my_eval_dataset.num_workers = self.args.dataloader_num_workers
        my_eval_dataset.drop_last = self.args.dataloader_drop_last
        my_eval_dataset.collate_batch = self.data_collator
        # ==================== end ==========================


        eval_dataloader = my_eval_dataset
        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        # for i in range(len(eval_dataloader)):
        #     srclen = 0; 
        #     while(eval_dataloader[i][1][srclen]==-100):srclen+=1
        #     inputs = eval_dataloader[i][0][0:srclen].unsqueeze(0)
        #     append_len = srclen + self.model.preseqlen
        #     output_sequences = self.model.generate(
        #         input_ids=inputs,
        #         # past_key_values=prompt,
        #         maxlen=100+append_len,
        #         temperature=0.5,
        #         tokenizer = self.tokenizer,
        #         decode_strategy = 'top-p',
        #         top_k=50267,
        #         gpt2 = self.gpt2,
        #         top_p=0.9,
        #     )
        #     tokenizer = self.tokenizer
        #     for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #         generated_sequence = generated_sequence.tolist()
        #         text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        #         text_output = text[len(tokenizer.decode(append_len, clean_up_tokenization_spaces=True)):]
        #         idx = text_output.find(tokenizer.eos_token)
        #         if idx >= 0:
        #             text_output = text_output[:idx]
        #         text_output = text_output.strip()
        #         print(text_output)


        self.log(output.metrics)


        return output.metrics



    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed.

        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """

        test_dataset.batch_size = self.args.eval_batch_size
        test_dataset.sampler = self._get_eval_sampler(eval_dataset=test_dataset)
        test_dataset.collate_batch = self.data_collator
        test_dataset.drop_last = self.args.dataloader_drop_last


        test_dataloader = test_dataset

        return self.prediction_loop(test_dataloader, description="Prediction")

    def prediction_loop(
        self, dataloader: Dataset, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: jt.Var = None
        label_ids: jt.Var = None
        entropy_losses: List[float] = []
        model.eval()
        if self.gpt2 is not None:
            self.gpt2.eval()

        tqdm.write(str(model.is_training()))
        tqdm.write(str(self.gpt2.is_training()))


        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)
            if logits is not None:
                preds = logits if preds is None else nested_concat(preds, logits, dim=0)
                temp_logits = [nn.log_softmax(x) for x in logits]
                entropy_losses.extend([(x.exp() * x).sum() for x in temp_logits])
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                    .mean()
                    .item()
                )
            else:
                metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        if len(entropy_losses) > 0:
            metrics['entropy'] = np.mean(entropy_losses)
            tqdm.write('entropy', metrics['entropy'] )

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[jt.Var, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[jt.Var], Optional[jt.Var]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        # At eval time, set the weights to 1/bsz. and see the results..

        # if 'weights' in inputs:
        #     weights = inputs['weights']
        #     bsz = weights.view(-1).shape[0]
        #     weights = (torch.ones(weights.shape)/bsz).to(weights.device)
        #     inputs['weights'] = weights

        with jt.no_grad():
            # outputs = model.forward_weighted(**inputs)
            outputs = model(**inputs, gpt2_model=self.gpt2)
            if has_labels:
                # The .mean() is to reduce in case of distributed training
                loss = outputs[0].mean().item()
                logits = outputs[1:]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = tuple(logit.detach() for logit in logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)

    def floating_point_ops(self, inputs: Dict[str, Union[jt.Var, Any]]):
        """
        For models that inherit from :class:`~transformers.PretrainedModel`, uses
        that method to compute the number of floating point operations for every backward + forward pass. If using
        another model, either implement such a method in the model or subclass and override this method.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """


        model = self.model

        if hasattr(model, "floating_point_ops"):
            return model.floating_point_ops(inputs)

        else:
            return 0
