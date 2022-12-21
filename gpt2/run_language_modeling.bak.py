# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from train_control import PrefixTuning, PrefixEmbTuning, GPT2LMHeadModel
from transformers.file_utils import cached_path
import jittor as jt
from jittor import init
from jittor import nn
import glob
from .utils.tokenization_utils import PreTrainedTokenizer

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig, # 无需修改
    AutoModelWithLMHead, # 无需修改
    AutoTokenizer,  # 无需修改
    DataCollatorForData2TextLanguageModeling, # 需修改
    HfArgumentParser, # 无需修改
    LineByLineData2TextTextDataset, # 需修改
    LineByLineWebNLGTextDataset, # 需修改
    PreTrainedTokenizer, # 无需修改
    TextDataset, # 需修改
    TrainingArguments, # 需修改
)

from trainer_prefix import Trainer_Prefix



logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    prefixModel_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The prefix model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch."
        },
    )

    prefix_mode: Optional[str] = field(
        default='activation',
        metadata={
            "help": "activation or embedding"
        },
    )



    preseqlen: Optional[int] = field(
        default=0,
        metadata={
            "help": "preseqlen for how many tokens of prefix should we include."
        },
    )

    optim_prefix: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether we are optimizing the prefix directly, or optimize another amortized function that "
                    "genrate the prefix."
        },
    )



    tuning_mode: Optional[str] = field(
        default='finetune',
        metadata={
            "help": "whether it's doing prefixtune or finetune."
        },
    )

    objective_mode: Optional[int] = field(
        default=2,
        metadata={
            "help": "In prefixtuning setting, the objective function... "
        },
    )

    top_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "In finetuning setting, if we only tune the top k layers. "
        },
    )

    adapter_design: Optional[int] = field(
        default=2,
        metadata={
            "help": "For Baseline of the adapter module... (1) means using the NLG adapter reference. "
                    "(2) means using a design similar to adapter module"
        },
    )

    adapter_bottleneck: Optional[int] = field(
        default=100,
        metadata={
            "help": "For baseline adapter module: the mid dim of the adapter. "
        },
    )

    parametrize_emb: Optional[str] = field(
        default='MLP',
        metadata={
            "help": "MLP or Emb to parametrize when we optimize for the embeddings."
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "dropout rate for the prefix tuning model. "
        },
    )

    teacher_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "dropout rate for the teacher model. "
        },
    )


    init_random: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to init a random embedding, or use GPT2 embedding for the prefix tuning model. "
        },
    )

    init_shallow : Optional[str] = field(
        default='no',
        metadata={
            "help": "shallow is default to be no, because we add reparametrization trick. If shallow=yes, "
                    "then no reparametrization "
        },
    )

    init_shallow_word: Optional[str] = field(
        default='no',
        metadata={
            "help": "when init_shallow is yes, what word to use... "
        },
    )


    use_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    use_custom_teacher_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    mid_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "the mid dim."
        },
    )


    gumbel: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the gumbel softmax trick in training."
        },
    )

    replay_buffer: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the replay buffer in training."
        },
    )

    training_obj: Optional[int] = field(
        default=0,
        metadata={
            "help": "use a specified training objective"
        },
    )




    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The task mode"}
    )

    matching_objective: Optional[str] = field(
        default='kl', metadata={"help": "The distillation objective"}
    )

    distill: Optional[str] = field(
        default='no', metadata={"help": "yes/no"}
    )

    finetuned_model_path: Optional[str] = field(
        default="/u/scr/xlisali/contrast_LM/transformers/examples/full/full/webnlgfinetune_n_20_act_cat_b=6-e"
                "=10_d=0.0_u=no_lr=1e-05_w=0.0_s=101_r=n_m=512_earlystop", metadata={"help": "finetuned model path (teacher model)"}
    )



    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )

    lowdata_token: Optional[str] = field(
        default='summarize', metadata={"help": "The token to be prepended at initialization time. "}
    )

    use_lowdata_token: Optional[str] = field(
        default='yes', metadata={"help": "Whether we should use the lowdata token and pass it to the prefixTuning Model "
                                         "for the initialization trick.  "}
    )


    train_embs: Optional[str] = field(
        default='no', metadata={"help": "whether the train word embeddings"}
    )

    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )

    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )

    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )

    # controlprefix: Optional[str] = field(
    #     default="yes", metadata={"help": "The control mode"}
    # )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    training_args: TrainingArguments = None,
    finetune_mode: bool = False,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        print(args.task_mode)
        if args.task_mode == 'data2text':
            dataset = LineByLineData2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                 eos_tok=tokenizer.eos_token,
                                                 lowdata_token=args.lowdata_token if ('lowdata' in training_args.output_dir and finetune_mode) else None)

        elif args.task_mode == 'webnlg':
            dataset = LineByLineWebNLGTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        else:
            print("task mode false")
        return dataset
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    jt.misc.set_global_seed(training_args.seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    config._my_arg_tune_mode = model_args.tuning_mode

    # 0 means the regular token level objective, which is sum / output_len
    # 1 means the sentence level objective, which is sum
    # 2 means our buggy version which is sum/max_batch(input_len +output_len)
    # 3 means our buggy version which is sum/max_batch(output_len)
    # 4 means our buggy version which is sum/(input_len +output_len)
    config._objective_mode = model_args.objective_mode
    config._my_arg_task_mode = data_args.task_mode

    if model_args.tuning_mode in ['finetune', 'adaptertune', 'finetune-top']:
        print('objective is 0 because of finetune')
    elif model_args.tuning_mode == 'prefixtune':
        print('objective is {}'.format(config._objective_mode ))
    # Added by MX

    if model_args.model_name_or_path:
        print(config.return_dict)
        config.return_dict = True
        model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir,
        )

    # HERE
    # model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)


    ##############################################################
    ################# ADJUST TOKENIZER ###########################
    ##############################################################

    print(model_args.tuning_mode)
    print('adapting the size of the model embedding to include [PAD]')
    print('len(tokenizer) = ', len(tokenizer))
    num_added_tokens = tokenizer.add_special_tokens(
        {'pad_token': '[PAD]'})
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    print('len(tokenizer) = ', len(tokenizer))
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.bos_token, tokenizer.bos_token_id)

    if model_args.tuning_mode == 'prefixtune': # prefixtune
        for param in model.base_model.parameters():
            param.requires_grad = False
        gpt2 = model


        print('loading the prefix model from ', model_args.prefixModel_name_or_path)
        # print(bool(".ckpt" in model_args.prefixModel_name_or_path))
        if model_args.optim_prefix == 'yes':
            optim_prefix_bool = True
        elif model_args.optim_prefix == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        if model_args.prefixModel_name_or_path is not None:
            config2 = AutoConfig.from_pretrained(model_args.prefixModel_name_or_path, cache_dir=model_args.cache_dir)
            # print(config2)

            if model_args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                        model_args.prefixModel_name_or_path,
                        from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                        config=config2,
                        cache_dir=model_args.cache_dir,
                        model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                        use_infix=(data_args.format_mode == 'infix')
                    )

            elif model_args.prefix_mode == 'activation':

                model = PrefixTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix')
                )
            else:
                assert False, "invalid prefix mode"

        else:

            # should clone the config and construct it.
            config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            config_prefix._my_arg_tune_mode = model_args.tuning_mode
            config_prefix._my_arg_task_mode = data_args.task_mode
            config_prefix._my_arg_control = True
            config_prefix.train_weights = data_args.train_embs
            config_prefix.optim_prefix = optim_prefix_bool
            config_prefix.preseqlen = model_args.preseqlen
            config_prefix.use_infix = (data_args.format_mode == 'infix')
            config_prefix.format_mode = data_args.format_mode
            config_prefix.prefix_dropout = model_args.prefix_dropout
            config_prefix.vocab_size = len(tokenizer)
            config_prefix.lowdata = ('lowdata' in training_args.output_dir)
            if not config_prefix.lowdata:
                config_prefix.lowdata = ('datalevels' in training_args.output_dir and data_args.use_lowdata_token == 'yes')
            if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
                config_prefix.lowdata_token = tokenizer([data_args.lowdata_token],
                                                        add_prefix_space=True)['input_ids']  #return_tensors='np',
                print(data_args.lowdata_token)
                print(config_prefix.lowdata_token)

            # some extra stuff.
            config_prefix.init_random = model_args.init_random
            config_prefix.mid_dim = model_args.mid_dim
            config_prefix.init_shallow = model_args.init_shallow
            if config_prefix.init_shallow == 'yes':
                if model_args.init_shallow_word != 'no':
                    config_prefix.init_shallow_word = tokenizer([model_args.init_shallow_word],
                                                                add_prefix_space=True)['input_ids']  #return_tensors='np',
                else:
                    config_prefix.init_shallow_word = None
                print(model_args.init_shallow_word)
                print(config_prefix.init_shallow_word)





            print('training the prefix model from scratch. ')
            if model_args.prefix_mode == 'embedding':
                # specific parametrization for embedding.
                config_prefix.parametrize_emb = model_args.parametrize_emb
                model = PrefixEmbTuning(config_prefix, model_gpt2=gpt2)

            elif model_args.prefix_mode == 'activation':
                model = PrefixTuning(config_prefix, model_gpt2=gpt2)
            else:
                assert False, "invalid prefix mode"


        discri_labels = None

    ##############################################################
    #################LOADING DATASETS ###########################
    ##############################################################
    print("line_by_line", data_args.line_by_line)
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, training_args=training_args,
                    finetune_mode=(model_args.tuning_mode == 'finetune')) #if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                    training_args=training_args, finetune_mode=(model_args.tuning_mode == 'finetune') )
        if training_args.do_eval
        else None
    )
    if data_args.task_mode == 'data2text' or data_args.task_mode== 'triples' or data_args.task_mode == 'webnlg':
        print('FORMAT MODE IS ', data_args.format_mode)
        data_collator = DataCollatorForData2TextLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
            format_mode=data_args.format_mode
        )
    else:
        print("task mode error 2")

    if (model_args.tuning_mode == 'prefixtune'):

        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            model_gpt2=gpt2,
            args=training_args,
            prediction_loss_only=True,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            task_mode =data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes'),
        )

    # Training


    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        trainer.train(model_path=model_path)

        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # eval_output = trainer.evaluate()
        eval_output = trainer.evaluate(train_dataset)

        # perplexity = math.exp(eval_output["eval_loss"])
        perplexity = eval_output["eval_loss"]
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results

if __name__ == "__main__":

    main()
