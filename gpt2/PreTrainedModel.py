import re
import jittor as jt
from transformers import PretrainedConfig, GPT2Config, load_tf_weights_in_gpt2
from transformers.file_utils import TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_NAME, cached_path, hf_bucket_url, is_remote_url
from transformers.generation_utils import BeamHypotheses
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.utils import logging
from jittor import nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import os
import torch
from transformers.file_utils import ModelOutput
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

logger = logging.get_logger(__name__)

def select_index(metrix: jt.Var, index: jt.Var, dim: int = 0) -> jt.Var:
    return metrix[(slice(None),)*dim+(index,)]


def top_k_top_p_filtering(
    logits: jt.Var,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> jt.Var:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < jt.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(torch.tensor(logits.data()), descending=True)
        sorted_logits = jt.from_torch(sorted_logits)
        sorted_indices = jt.from_torch(sorted_indices)
        cumulative_probs = jt.cumsum(nn.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        # w = jt.empty((nx, nf))
        w = jt.normal(mean=0,std=0.2,size=(nx, nf))
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(jt.zeros(nf))

    def execute(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = self.bias + x.view(-1, x.size(-1)) @ self.weight
        x = x.view(*size_out)
        return x

def calc_banned_ngram_tokens(prev_input_ids: jt.Var, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens



def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores: jt.Var, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = np.array(banned_mask_list, dtype = np.long)
    indices = np.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]
    # print(banned_mask)
    # print(indices)
    banned_mask = jt.from_torch(torch.sparse.LongTensor(torch.Tensor(banned_mask.T).long(), torch.Tensor(indices), list(scores.size())).to_dense().bool())
    scores.masked_fill(banned_mask, -float("inf"))


# class GenerationMixin:
#     """
#     A class contraining all of the functions supporting generation, to be used as a mixin in
#     :class:`~transfomers.PreTrainedModel`.
#     """

#     def prepare_inputs_for_generation(self, input_ids, **kwargs):
#         """
#         Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
#         generate method.
#         """
#         return {"input_ids": input_ids}

#     def prepare_inputs_for_generation2(self, input_ids, **kwargs):
#         """
#         Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
#         generate method.
#         """
#         # print(kwargs)
#         return {"input_ids": input_ids, 'emb_match':kwargs['emb_match'],
#                 'control_code':kwargs['control_code'], 'past_key_values':kwargs['past_key_values']}

#     def adjust_logits_during_generation(self, logits, **kwargs):
#         """
#         Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to adjust the logits in
#         the generate method.
#         """
#         return logits

#     def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
#         """
#         Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
#         """
#         for i in range(batch_size * num_beams):
#             for previous_token in set(prev_output_tokens[i].tolist()):
#                 # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
#                 if lprobs[i, previous_token] < 0:
#                     lprobs[i, previous_token] *= repetition_penalty
#                 else:
#                     lprobs[i, previous_token] /= repetition_penalty

#     def postprocess_next_token_scores(
#         self,
#         scores,
#         input_ids,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         cur_len,
#         min_length,
#         max_length,
#         eos_token_id,
#         repetition_penalty,
#         batch_size,
#         num_beams,
#     ):
#         # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
#         if repetition_penalty != 1.0:
#             self.enforce_repetition_penalty_(
#                 scores,
#                 batch_size,
#                 num_beams,
#                 input_ids,
#                 repetition_penalty,
#             )

#         # set eos token prob to zero if min_length is not reached
#         if eos_token_id is not None and cur_len < min_length:
#             scores[:, eos_token_id] = -float("inf")

#         if no_repeat_ngram_size > 0:
#             # calculate a list of banned tokens to prevent repetitively generating the same ngrams
#             num_batch_hypotheses = batch_size * num_beams
#             # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
#             banned_batch_tokens = calc_banned_ngram_tokens(
#                 input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
#             )
#             for i, banned_tokens in enumerate(banned_batch_tokens):
#                 scores[i, banned_tokens] = -float("inf")

#         if bad_words_ids is not None:
#             # Exclude EOS token (already processed)
#             bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
#             # calculate a list of banned tokens according to bad words
#             banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
#             # print(bad_words_ids, banned_tokens)
#             # Modify the scores in place by setting the banned tokens logits to `-inf`
#             set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

#         return scores

#     def generate(
#         self,
#         input_ids: Optional[jt.Var] = None,
#         max_length: Optional[int] = None,
#         min_length: Optional[int] = None,
#         do_sample: Optional[bool] = None,
#         early_stopping: Optional[bool] = None,
#         num_beams: Optional[int] = None,
#         temperature: Optional[float] = None,
#         top_k: Optional[int] = None,
#         top_p: Optional[float] = None,
#         repetition_penalty: Optional[float] = None,
#         bad_words_ids: Optional[Iterable[int]] = None,
#         bos_token_id: Optional[int] = None,
#         pad_token_id: Optional[int] = None,
#         eos_token_id: Optional[int] = None,
#         length_penalty: Optional[float] = None,
#         no_repeat_ngram_size: Optional[int] = None,
#         num_return_sequences: Optional[int] = None,
#         attention_mask: Optional[jt.Var] = None,
#         decoder_start_token_id: Optional[int] = None,
#         use_cache: Optional[bool] = None,
#         **model_kwargs
#     ) -> jt.Var:
#         r"""
#         Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
#         beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

#         Adapted in part from `Facebook's XLM beam search code
#         <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

#         Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
#         attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
#         indicated are the default values of those config.

#         Most of these parameters are explained in more detail in `this blog post
#         <https://huggingface.co/blog/how-to-generate>`__.

#         Parameters:

#             input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 The sequence used as a prompt for the generation. If :obj:`None` the method initializes
#                 it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
#             max_length (:obj:`int`, `optional`, defaults to 20):
#                 The maximum length of the sequence to be generated.
#             min_length (:obj:`int`, `optional`, defaults to 10):
#                 The minimum length of the sequence to be generated.
#             do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether or not to use sampling ; use greedy decoding otherwise.
#             early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
#             num_beams (:obj:`int`, `optional`, defaults to 1):
#                 Number of beams for beam search. 1 means no beam search.
#             temperature (:obj:`float`, `optional`, defaults tp 1.0):
#                 The value used to module the next token probabilities.
#             top_k (:obj:`int`, `optional`, defaults to 50):
#                 The number of highest probability vocabulary tokens to keep for top-k-filtering.
#             top_p (:obj:`float`, `optional`, defaults to 1.0):
#                 If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
#                 higher are kept for generation.
#             repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
#                 The parameter for repetition penalty. 1.0 means no penalty. See `this paper
#                 <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
#             pad_token_id (:obj:`int`, `optional`):
#                 The id of the `padding` token.
#             bos_token_id (:obj:`int`, `optional`):
#                 The id of the `beginning-of-sequence` token.
#             eos_token_id (:obj:`int`, `optional`):
#                 The id of the `end-of-sequence` token.
#             length_penalty (:obj:`float`, `optional`, defaults to 1.0):
#                 Exponential penalty to the length. 1.0 means no penalty.

#                 Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
#                 order to encourage the model to produce longer sequences.
#             no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
#                 If set to int > 0, all ngrams of that size can only occur once.
#             bad_words_ids(:obj:`List[int]`, `optional`):
#                 List of token ids that are not allowed to be generated. In order to get the tokens of the words that
#                 should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
#             num_return_sequences(:obj:`int`, `optional`, defaults to 1):
#                 The number of independently computed returned sequences for each element in the batch.
#             attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
#                 tokens that are not masked, and 0 for masked tokens.

#                 If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

#                 `What are attention masks? <../glossary.html#attention-mask>`__
#             decoder_start_token_id (:obj:`int`, `optional`):
#                 If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
#             use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
#                 Whether or not the model should use the past last key/values attentions (if applicable to the model) to
#                 speed up decoding.
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

#         Return:

#             :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
#             The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
#             shorter if all batches finished early due to the :obj:`eos_token_id`.

#         Examples::

#             tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
#             outputs = model.generate(max_length=40)  # do greedy decoding
#             print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
#             input_context = 'The dog'
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
#             for i in range(3): #  3 output sequences were generated
#                 print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
#             input_context = 'The dog'
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
#             for i in range(3): #  3 output sequences were generated
#                 print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
#             input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
#             print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

#             tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
#             model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
#             input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
#             bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
#             input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
#             outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
#         """

#         # We cannot generate if the model does not have a LM head
#         if self.get_output_embeddings() is None:
#             raise AttributeError(
#                 "You tried to generate sequences with a model that does not have a LM Head."
#                 "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
#             )

#         max_length = max_length if max_length is not None else self.config.max_length
#         min_length = min_length if min_length is not None else self.config.min_length
#         do_sample = do_sample if do_sample is not None else self.config.do_sample
#         early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         num_beams = num_beams if num_beams is not None else self.config.num_beams
#         temperature = temperature if temperature is not None else self.config.temperature
#         top_k = top_k if top_k is not None else self.config.top_k
#         top_p = top_p if top_p is not None else self.config.top_p
#         repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
#         bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
#         pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
#         length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
#         no_repeat_ngram_size = (
#             no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
#         )
#         bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
#         num_return_sequences = (
#             num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
#         )
#         decoder_start_token_id = (
#             decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
#         )

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]  # overriden by the input batch_size
#         else:
#             batch_size = 1

#         assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
#         assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
#         assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
#         assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
#         assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
#         assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
#         assert temperature > 0, "`temperature` should be strictly positive."
#         assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
#         assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
#         assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
#         assert input_ids is not None or (
#             isinstance(bos_token_id, int) and bos_token_id >= 0
#         ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
#         assert pad_token_id is None or (
#             isinstance(pad_token_id, int) and (pad_token_id >= 0)
#         ), "`pad_token_id` should be a positive integer."
#         assert (eos_token_id is None) or (
#             isinstance(eos_token_id, int) and (eos_token_id >= 0)
#         ), "`eos_token_id` should be a positive integer."
#         assert length_penalty > 0, "`length_penalty` should be strictly positive."
#         assert (
#             isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
#         ), "`no_repeat_ngram_size` should be a positive integer."
#         assert (
#             isinstance(num_return_sequences, int) and num_return_sequences > 0
#         ), "`num_return_sequences` should be a strictly positive integer."
#         assert (
#             bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
#         ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

#         if input_ids is None:
#             assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
#                 "you should either supply a context to complete as `input_ids` input "
#                 "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
#             )
#             input_ids = jt.full(
#                 (batch_size, 1),
#                 bos_token_id
#             ).long()
#         else:
#             assert input_ids.ndim == 2, "Input prompt should be of shape (batch_size, sequence length)."

#         # not allow to duplicate outputs when greedy decoding
#         if do_sample is False:
#             if num_beams == 1:
#                 # no_beam_search greedy generation conditions
#                 assert (
#                     num_return_sequences == 1
#                 ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

#             else:
#                 # beam_search greedy generation conditions
#                 assert (
#                     num_beams >= num_return_sequences
#                 ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

#         # create attention mask if necessary
#         # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
#         if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
#             attention_mask = (input_ids!=pad_token_id).long()
#         elif attention_mask is None:
#             attention_mask = jt.ones(input_ids.shape, dtype = input_ids.dtype)

#         # set pad_token_id to eos_token_id if not set. Important that this is done after
#         # attention_mask is created
#         if pad_token_id is None and eos_token_id is not None:
#             logger.warning(
#                 "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
#             )
#             pad_token_id = eos_token_id

#         # vocab size
#         if hasattr(self.config, "vocab_size"):
#             vocab_size = self.config.vocab_size
#         elif (
#             self.config.is_encoder_decoder
#             and hasattr(self.config, "decoder")
#             and hasattr(self.config.decoder, "vocab_size")
#         ):
#             vocab_size = self.config.decoder.vocab_size
#         else:
#             raise ValueError("either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined")

#         # set effective batch size and effective batch multiplier according to do_sample
#         if do_sample:
#             effective_batch_size = batch_size * num_return_sequences
#             effective_batch_mult = num_return_sequences
#         else:
#             effective_batch_size = batch_size
#             effective_batch_mult = 1

#         if self.config.is_encoder_decoder:
#             if decoder_start_token_id is None:
#                 # see if BOS token can be used for decoder_start_token_id
#                 if bos_token_id is not None:
#                     decoder_start_token_id = bos_token_id
#                 elif (
#                     hasattr(self.config, "decoder")
#                     and hasattr(self.config.decoder, "bos_token_id")
#                     and self.config.decoder.bos_token_id is not None
#                 ):
#                     decoder_start_token_id = self.config.decoder.bos_token_id
#                 else:
#                     raise ValueError(
#                         "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
#                     )

#             assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
#             assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

#             # get encoder and store encoder outputs
#             encoder = self.get_encoder()
#             # LISA CHANGED HERE
#             temp_past_key_values = model_kwargs['past_key_values'] if 'past_key_values' in model_kwargs else None
#             encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask,
#                                                    past_key_values=temp_past_key_values, return_dict=True)
#             # print(temp_past_key_values[0].keys())
#             # # LISA remove the encoder part.
#             # if 'encoder' in temp_past_key_values[0].keys():
#             #     for i in range(len(temp_past_key_values)):
#             #         del temp_past_key_values[i]['encoder']


#         # Expand input ids if num_beams > 1 or num_return_sequences > 1
#         if num_return_sequences > 1 or num_beams > 1:
#             input_ids_len = input_ids.shape[-1]
#             input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
#             #URGENT
#             attn_len = attention_mask.shape[-1]
#             attention_mask = attention_mask.unsqueeze(1).expand(
#                 batch_size, effective_batch_mult * num_beams, attn_len
#             )

#             # attention_mask = attention_mask.unsqueeze(1).expand(
#             #     batch_size, effective_batch_mult * num_beams, input_ids_len
#             # )

#             input_ids = input_ids.contiguous().view(
#                 effective_batch_size * num_beams, input_ids_len
#             )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

#             #URGENT
#             attention_mask = attention_mask.contiguous().view(
#                 effective_batch_size * num_beams, attn_len
#             )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

#             # attention_mask = attention_mask.contiguous().view(
#             #     effective_batch_size * num_beams, input_ids_len
#             # )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

#         if self.config.is_encoder_decoder:
#             # create empty decoder input_ids
#             input_ids = jt.full(
#                 (effective_batch_size * num_beams, 1),
#                 decoder_start_token_id,
#             ).long()
#             cur_len = 1

#             assert (
#                 batch_size == encoder_outputs.last_hidden_state.shape[0]
#             ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

#             # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
#             expanded_batch_idxs = (
#                 jt.arange(batch_size)
#                 .view(-1, 1)
#                 .repeat(1, num_beams * effective_batch_mult)
#                 .view(-1)
#             )

#             # expand encoder_outputs
#             encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
#                 0, expanded_batch_idxs
#             )

#             # save encoder_outputs in `model_kwargs`
#             model_kwargs["encoder_outputs"] = encoder_outputs

#         else:
#             cur_len = input_ids.shape[-1]

#         assert (
#             cur_len < max_length
#         ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

#         if num_beams > 1:
#             output = self._generate_beam_search(
#                 input_ids,
#                 cur_len=cur_len,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=do_sample,
#                 early_stopping=early_stopping,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id,
#                 batch_size=effective_batch_size,
#                 num_return_sequences=num_return_sequences,
#                 length_penalty=length_penalty,
#                 num_beams=num_beams,
#                 vocab_size=vocab_size,
#                 attention_mask=attention_mask,
#                 use_cache=use_cache,
#                 model_kwargs=model_kwargs,
#             )
#         else:
#             output = self._generate_no_beam_search(
#                 input_ids,
#                 cur_len=cur_len,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=do_sample,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id,
#                 batch_size=effective_batch_size,
#                 attention_mask=attention_mask,
#                 use_cache=use_cache,
#                 model_kwargs=model_kwargs,
#             )

#         return output

#     def _generate_no_beam_search(
#         self,
#         input_ids,
#         cur_len,
#         max_length,
#         min_length,
#         do_sample,
#         temperature,
#         top_k,
#         top_p,
#         repetition_penalty,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         pad_token_id,
#         eos_token_id,
#         batch_size,
#         attention_mask,
#         use_cache,
#         model_kwargs,
#     ):
#         """Generate sequences for each example without beam search (num_beams == 1).
#         All returned sequence are generated independantly.
#         """
#         # length of generated sentences / unfinished sentences
#         unfinished_sents = jt.ones(batch_size,dtype=input_ids.dtype)
#         sent_lengths = jt.ones(batch_size,dtype=input_ids.dtype)*max_length

#         past = None
#         # print('line 533')
#         while cur_len < max_length:

#             # print(past)
#             # if past is not None:
#             #     print(past[0].shape, len(past))
#             # else:
#             #     print(past)

#             # model_inputs = self.prepare_inputs_for_generation2(
#             #     input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
#             # )

#             model_inputs = self.prepare_inputs_for_generation(
#                 input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
#             )

#             outputs = self(**model_inputs, return_dict=True)
#             next_token_logits = outputs.logits[:, -1, :]

#             scores = self.postprocess_next_token_scores(
#                 scores=next_token_logits,
#                 input_ids=input_ids,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 cur_len=cur_len,
#                 min_length=min_length,
#                 max_length=max_length,
#                 eos_token_id=eos_token_id,
#                 repetition_penalty=repetition_penalty,
#                 batch_size=batch_size,
#                 num_beams=1,
#             )

#             # if model has past, then set the past variable to speed up decoding
#             if "past_key_values" in outputs:
#                 past = outputs.past_key_values
#             elif "mems" in outputs:
#                 past = outputs.mems


#             if do_sample:
#                 # Temperature (higher temperature => more likely to sample low probability tokens)
#                 if temperature != 1.0:
#                     scores = scores / temperature
#                 # Top-p/top-k filtering
#                 next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
#                 # Sample
#                 probs = nn.softmax(next_token_logscores, dim=-1)
#                 next_token = jt.from_torch(torch.multinomial(probs, num_samples=1).squeeze(1))
#             else:
#                 # Greedy decoding
#                 next_token = jt.argmax(next_token_logits, dim=-1)

#             # update generations and finished sentences
#             if eos_token_id is not None:
#                 # pad finished sentences if eos_token_id exist
#                 tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
#             else:
#                 tokens_to_add = next_token

#             # add token and increase length by one
#             input_ids = jt.concat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
#             cur_len = cur_len + 1

#             if eos_token_id is not None:
#                 eos_in_sents = tokens_to_add == eos_token_id
#                 # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
#                 is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
#                 sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
#                 # unfinished_sents is set to zero if eos in sentence
#                 unfinished_sents.mul_((~eos_in_sents).long())

#             # stop when there is a </s> in each sentence, or if we exceed the maximul length
#             if unfinished_sents.max() == 0:
#                 break

#             # extend attention_mask for new generated input if only decoder
#             if self.config.is_encoder_decoder is False:
#                 attention_mask = jt.concat(
#                     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
#                 )

#         return input_ids

#     def _generate_beam_search(
#         self,
#         input_ids,
#         cur_len,
#         max_length,
#         min_length,
#         do_sample,
#         early_stopping,
#         temperature,
#         top_k,
#         top_p,
#         repetition_penalty,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         pad_token_id,
#         eos_token_id,
#         batch_size,
#         num_return_sequences,
#         length_penalty,
#         num_beams,
#         vocab_size,
#         attention_mask,
#         use_cache,
#         model_kwargs,
#     ):
#         """Generate sequences for each example with beam search."""

#         # generated hypotheses
#         generated_hyps = [
#             BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
#             for _ in range(batch_size)
#         ]

#         # scores for each sentence in the beam
#         beam_scores = jt.zeros((batch_size, num_beams)).float()

#         # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
#         if do_sample is False:
#             beam_scores[:, 1:] = -1e9
#         beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

#         # cache compute states
#         past = None

#         # done sentences
#         done = [False for _ in range(batch_size)]

#         while cur_len < max_length:
#             model_inputs = self.prepare_inputs_for_generation(
#                 input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
#             )
#             outputs = self(**model_inputs, return_dict=True)  # (batch_size * num_beams, cur_len, vocab_size)
#             next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

#             # if model has past, then set the past variable to speed up decoding
#             if "past_key_values" in outputs:
#                 past = outputs.past_key_values
#             elif "mems" in outputs:
#                 past = outputs.mems

#             if self.config.is_encoder_decoder and do_sample is False:
#                 # TODO (PVP) still a bit hacky here - there might be a better solution
#                 next_token_logits = self.adjust_logits_during_generation(
#                     next_token_logits, cur_len=cur_len, max_length=max_length
#                 )

#             scores = nn.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

#             scores = self.postprocess_next_token_scores(
#                 scores=scores,
#                 input_ids=input_ids,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 cur_len=cur_len,
#                 min_length=min_length,
#                 max_length=max_length,
#                 eos_token_id=eos_token_id,
#                 repetition_penalty=repetition_penalty,
#                 batch_size=batch_size,
#                 num_beams=num_beams,
#             )

#             assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
#                 scores.shape, (batch_size * num_beams, vocab_size)
#             )

#             if do_sample:
#                 _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
#                 # Temperature
#                 if temperature != 1.0:
#                     _scores = _scores / temperature
#                 # Top-p/top-k filtering
#                 _scores = top_k_top_p_filtering(
#                     _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
#                 )  # (batch_size * num_beams, vocab_size)
#                 # re-organize to group the beam together to sample from all beam_idxs
#                 _scores = _scores.contiguous().view(
#                     batch_size, num_beams * vocab_size
#                 )  # (batch_size, num_beams * vocab_size)

#                 # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
#                 probs = nn.softmax(_scores, dim=-1)
#                 next_tokens = jt.from_torch(torch.multinomial(probs, num_samples=2 * num_beams))  # (batch_size, num_beams * 2)
#                 # Compute next scores
#                 next_scores = jt.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
#                 # sort the sampled vector to make sure that the first num_beams samples are the best
#                 next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
#                 next_scores = jt.from_torch(next_scores)
#                 next_scores_indices = jt.from_torch(next_scores_indices)
#                 next_tokens = jt.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

#             else:
#                 next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

#                 # re-organize to group the beam together (we are keeping top hypothesis accross beams)
#                 next_scores = next_scores.view(
#                     batch_size, num_beams * vocab_size
#                 )  # (batch_size, num_beams * vocab_size)

#                 next_scores, next_tokens = jt.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

#             assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

#             # next batch beam content
#             next_batch_beam = []

#             # for each sentence
#             for batch_idx in range(batch_size):

#                 # if we are done with this sentence, add a pad token
#                 if done[batch_idx]:
#                     assert (
#                         len(generated_hyps[batch_idx]) >= num_beams
#                     ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
#                     assert (
#                         eos_token_id is not None and pad_token_id is not None
#                     ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
#                     next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
#                     continue

#                 # next sentence beam content, this will get added to next_batch_beam
#                 next_sent_beam = []

#                 # next tokens for this sentence
#                 for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
#                     zip(next_tokens[batch_idx], next_scores[batch_idx])
#                 ):
#                     # get beam and token IDs
#                     beam_id = beam_token_id // vocab_size
#                     token_id = beam_token_id % vocab_size


#                     effective_beam_id = batch_idx * num_beams + beam_id
#                     # add to generated hypotheses if end of sentence
#                     if (eos_token_id is not None) and (token_id.item() == eos_token_id):
#                         # if beam_token does not belong to top num_beams tokens, it should not be added
#                         is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
#                         if is_beam_token_worse_than_top_num_beams:
#                             continue
#                         generated_hyps[batch_idx].add(
#                             input_ids[effective_beam_id].clone(),
#                             beam_token_score.item(),
#                         )
#                     else:
#                         # add next predicted token since it is not eos_token
#                         next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

#                     # once the beam for next step is full, don't add more tokens to it.
#                     if len(next_sent_beam) == num_beams:
#                         break

#                 # Check if we are done so that we can save a pad step if all(done)
#                 done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
#                     next_scores[batch_idx].max().item(), cur_len
#                 )

#                 # update next beam content
#                 assert len(next_sent_beam) == num_beams, "Beam should always be full"
#                 next_batch_beam.extend(next_sent_beam)
#                 assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

#             # stop when we are done with each sentence
#             if all(done):
#                 break

#             # sanity check / prepare next batch
#             assert len(next_batch_beam) == batch_size * num_beams
#             beam_scores = jt.array([x[0].item() for x in next_batch_beam] , dtype = beam_scores.dtype)
#             beam_tokens = jt.array([x[1].item() for x in next_batch_beam] , dtype = input_ids.dtype)
#             beam_idx = jt.array([x[2].item() for x in next_batch_beam], dtype = input_ids.dtype)

#             # re-order batch and update current length
#             input_ids = input_ids[beam_idx, :]
#             input_ids = jt.concat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
#             cur_len = cur_len + 1

#             # re-order internal states
#             if past is not None:
#                 past = self._reorder_cache(past, beam_idx)

#             # extend attention_mask for new generated input if only decoder
#             if self.config.is_encoder_decoder is False:
#                 attention_mask = jt.concat(
#                     [attention_mask, jt.ones((attention_mask.shape[0], 1), dtype = attention_mask.dtype)], dim=-1
#                 )

#         # finalize all open beam hypotheses and add to generated hypotheses
#         for batch_idx in range(batch_size):
#             if done[batch_idx]:
#                 continue

#             # test that beam scores match previously calculated scores if not eos and batch_idx not done
#             if eos_token_id is not None and all(
#                 (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
#             ):
#                 assert jt.all(
#                     next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
#                 ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
#                     next_scores[:, :num_beams][batch_idx],
#                     beam_scores.view(batch_size, num_beams)[batch_idx],
#                 )

#             # need to add best num_beams hypotheses to generated hyps
#             for beam_id in range(num_beams):
#                 effective_beam_id = batch_idx * num_beams + beam_id
#                 final_score = beam_scores[effective_beam_id].item()
#                 final_tokens = input_ids[effective_beam_id]
#                 generated_hyps[batch_idx].add(final_tokens, final_score)

#         # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
#         output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
#         output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

#         # select the best hypotheses
#         sent_lengths = jt.zeros(output_batch_size, dtype = input_ids.dtype)
#         best = []

#         # retrieve best hypotheses
#         for i, hypotheses in enumerate(generated_hyps):
#             sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
#             for j in range(output_num_return_sequences_per_batch):
#                 effective_batch_idx = output_num_return_sequences_per_batch * i + j
#                 best_hyp = sorted_hyps.pop()[1]
#                 sent_lengths[effective_batch_idx] = len(best_hyp)
#                 best.append(best_hyp)

#         # prepare for adding eos
#         sent_max_len = min(sent_lengths.max().item() + 1, max_length)
#         decoded = jt.zeros((output_batch_size, sent_max_len), dtype = input_ids.dtype)
#         # shorter batches are padded if needed
#         if sent_lengths.min().item() != sent_lengths.max().item():
#             assert pad_token_id is not None, "`pad_token_id` has to be defined"
#             decoded.fill_(pad_token_id)

#         # fill with hypotheses and eos_token_id if the latter fits in
#         for i, hypo in enumerate(best):
#             # print(sent_lengths[i])
#             decoded[i, : sent_lengths[i].item()] = hypo
#             if sent_lengths[i].item() < max_length:
#                 decoded[i, sent_lengths[i].item()] = eos_token_id

#         return decoded

#     @staticmethod
#     def _reorder_cache(past: Tuple, beam_idx: jt.Var) -> Tuple[jt.Var]:
#         return tuple(select_index(layer_past, beam_idx, 1) for layer_past in past)




class PreTrainedModel(nn.Module, ModuleUtilsMixin):
    r"""
    Base class for all models.

    :class:`~transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods
    for loading, downloading and saving models as well as a few methods common to all models to:

        * resize the input embeddings,
        * prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):
        - **config_class** (:class:`~transformers.PretrainedConfig`) -- A subclass of
          :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
        - **load_tf_weights** (:obj:`Callable`) -- A python `method` for loading a TensorFlow checkpoint in a
          PyTorch model, taking as arguments:

            - **model** (:class:`~transformers.PreTrainedModel`) -- An instance of the model on which to load the
              TensorFlow checkpoint.
            - **config** (:class:`~transformers.PreTrainedConfig`) -- An instance of the configuration associated
              to the model.
            - **path** (:obj:`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (:obj:`str`) -- A string indicating the attribute associated to the base model in
          derived classes of the same architecture adding modules on top of the base model.
        - **authorized_missing_keys** (:obj:`Optional[List[str]]`) -- A list of re pattern of tensor names to ignore
          when loading the model (and avoid unnecessary warnings).
        - **keys_to_never_save** (:obj:`Optional[List[str]]`) -- A list of of tensor names to ignore
          when saving the model (useful for keys that aren't trained, but which are deterministic)

    """
    config_class = None
    base_model_prefix = ""
    authorized_missing_keys = None
    authorized_unexpected_keys = None
    keys_to_never_save = None

    @property
    def dummy_inputs(self) -> Dict[str, jt.Var]:
        """
        :obj:`Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": jt.Var(DUMMY_INPUTS)}

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    @property
    def base_model(self) -> nn.Module:
        """
        :obj:`torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        """
        Returns the model's output embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

    @staticmethod
    def _tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str):
        uninitialized_encoder_weights: List[str] = []
        assert decoder.__class__ == encoder.__class__, f"{decoder.__class__} and {encoder.__class__} have to be equal."

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and substract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a :obj:`tie_weights()` method.

        Arguments:
            new_num_tokens (:obj:`int`, `optional`):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or :obj:`None`,
                just returns a pointer to the input tokens :obj:`torch.nn.Embedding` module of the model wihtout doing
                anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model wihtout doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # Tie weights if needed
        self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list
                of heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will
                prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self.keys_to_never_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self.keys_to_never_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        # if getattr(self.config, "xla_device", False):
        #     import jittor_xla.core.xla_model as xm

        #     if xm.is_master_ordinal():
        #         # Save configuration file
        #         model_to_save.config.save_pretrained(save_directory)
        #     # xm.save takes care of saving only from master
        #     xm.save(state_dict, output_model_file)
        # else:
        model_to_save.config.save_pretrained(save_directory)
        jt.save(state_dict, output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated).
        To train the model, you should first set it back in training mode with ``model.train()``.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str`, `optional`):
                Can be either:

                    - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                      ``bert-base-uncased``.
                    - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                      ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `shortcut name` string of a
                      pretrained model).
                    - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by suppling the save directory.
                    - The model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~transformers.PreTrainedModel.save_pretrained` and
                :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each
                request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionnary containing missing keys, unexpected keys and error
                messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try doanloading the model).
            use_cdn(:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to use Cloudfront (a Content Delivery Network, or CDN) when searching for the model on
                our S3 (faster). Should be set to :obj:`False` for checkpoints larger than 20GB.
            mirror(:obj:`str`, `optional`, defaults to :obj:`None`):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility problem,
                you can set this option to resolve it. Note that we do not guarantee the timeliness or safety. Please
                refer to the mirror site for more information.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            >>> from transformers import BertConfig, BertModel
            >>> # Download model and configuration from S3 and cache.
            >>> model = BertModel.from_pretrained('bert-base-uncased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = BertModel.from_pretrained('./test/saved_model/')
            >>> # Update configuration during loading.
            >>> model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)
        mirror = kwargs.pop("mirror", None)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if  os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif  os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, 'pytorch_model.pkl')):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.pkl')
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, 'pytorch_model.pth')):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.pth')
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            from torch import load as t_load
            try:
                state_dict = t_load(resolved_archive_file)
            except RuntimeError:
                import pickle
                with open(resolved_archive_file, 'rb') as f:
                    obj = f.read()
                    state_dict = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}


        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            # def load(module: nn.Module):
            #     module.load_state_dict(
            #         state_dict
            #     )
                # for name, child in module.named_modules()[1:]:
                #     if child is not None:
                #         load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            # load(model_to_load)
            model_to_load.load_state_dict(state_dict)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]
                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            # Some models may have keys that are not in the state by design, removing them before needlessly warning
            # the user.
            if cls.authorized_missing_keys is not None:
                for pat in cls.authorized_missing_keys:
                    missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

            if cls.authorized_unexpected_keys is not None:
                for pat in cls.authorized_unexpected_keys:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                    f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                    f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
                    f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                    f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                    f"and are newly initialized: {missing_keys}\n"
                    f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
                )
            else:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                    f"If your task is similar to the task the model of the checkpoint was trained on, "
                    f"you can already use {model.__class__.__name__} for predictions without further training."
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info


        return model


class GPT2PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight = jt.normal(mean=0.0, std=self.config.initializer_range, size=module.weight.size(),dtype=module.weight.dtype)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias = jt.zeros(module.bias.size(),dtype=module.bias.dtype)
        elif isinstance(module, nn.LayerNorm):
            module.bias = jt.zeros(module.bias.size(),dtype = module.bias.dtype)
            module.weight = jt.ones(module.bias.size(),dtype = module.weight.dtype)
