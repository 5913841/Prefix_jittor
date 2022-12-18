import sys
import logging
import os
import copy
from jittor.dataset import Dataset
import jittor as jt
import json
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple

sys.path.insert(0, '../')

from transformers import PreTrainedTokenizer, BatchEncoding

logger = logging.get_logger(__name__)


class LineByLineData2TextTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, lowdata_token:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if lowdata_token is None:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} {} '.format(lowdata_token, src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx-1
                temp_tgt_len += len(elem) - (sep_idx-1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (jt.array(self.examples[i]).long(),
                jt.array(self.labels[i]).long(),
                jt.array(self.src_sent[i]).long(),
                jt.array(self.tgt_sent[i]).long(),
                jt.array(self.src_cat[i]).long())
     
   
class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)



        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        print()
        print(self.labels[1])
        print(self.examples[1])
        print(edited_sents[1])
        print(self.src_sent[1])
        print(self.tgt_sent[1])
        print(self.src_cat[1])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (jt.Var(self.examples[i]).long(),
                jt.Var(self.labels[i]).long(),
                jt.Var(self.src_sent[i]).long(),
                jt.Var(self.tgt_sent[i]).long(),
                jt.Var(self.src_cat[i]).long(),
                )
        
        
def pad_sequence(sequences: List[jt.Var], batch_first=False, padding_val:float = 0):
    max_batch_size = -1
    max_shape = None
    # 获取最大的batch_size
    for var in sequences:
        if (var.size()[0] > max_batch_size):
            max_batch_size = var.size()[0]
            max_shape = jt.full_like(var, padding_val)
     
    for i, var in enumerate(sequences):
        tmp = max_shape.clone()
        tmp[:var.size()[0]] = var
        sequences[i] = tmp.clone()
            
            
    stacked = jt.stack(sequences)
    if (batch_first == False):
        stacked = stacked.transpose(0, 1)
    return stacked

@dataclass
class DataCollatorForData2TextLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], jt.Var, Dict[str, jt.Var]]]
    ) -> Dict[str, jt.Var]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        input_ids, labels, src, tgt, cate = zip(*examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:

            # print(self.format_mode)
            if self.format_mode == 'cat':
                mode_input = 3
            elif self.format_mode == 'peek':
                mode_input = 1
            elif self.format_mode == 'nopeek':
                mode_input = 2
            elif self.format_mode == 'infix':
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            # print(self.format_mode, mode_input)

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
                # tgt = self._tensorize_batch(tgt)
            elif mode_input == 2:
                # nopeek.
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
            elif mode_input == 3:
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(cate)
                cate_batch, cate_attn = None, None
            elif mode_input == 4:
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)

                cate_batch = self._tensorize_batch(cate)
                cate_attn = (cate_batch != self.tokenizer.pad_token_id)

            labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
            src_attn = (src != self.tokenizer.pad_token_id) # src
            tgt_attn = (batch != self.tokenizer.pad_token_id) # tgt

            if cate_batch is None:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn,
                        'src':src}
            else:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn': tgt_attn,
                        'src': src, "cate_batch":cate_batch, "cate_attn":cate_attn}

    def _tensorize_batch(
        self, examples: List[Union[List[int], jt.Var, Dict[str, jt.Var]]]
    ) -> jt.Var:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [jt.Var(e).long() for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return jt.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = jt.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(jt.array(special_tokens_mask).bool(), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = jt.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = jt.bernoulli(jt.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = jt.bernoulli(jt.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = jt.randint(len(self.tokenizer), labels.shape).long()
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels