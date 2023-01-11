# from transformers import Trainer
from typing import Optional, Tuple, Union
import jittor as jt
from pretrainedmodel import PreTrainedModel
from jittor import nn
import numpy as np

class PrefixTuning(PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, preseqlen=5):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.num_hidden_layers
        self.n_embd = config.hidden_size
        self.mid_dim = 512

        if hasattr(config, 'preseqlen'):
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen


        print('PrefixTuning')
        print('preseqlen is {}, optimizing the prefix directly'.format(self.preseqlen))
        self.input_tokens = jt.arange(self.preseqlen * self.match_n_layer).long().stop_grad()
        # self.wte = nn.Embedding(self.preseqlen * self.match_n_layer, self.n_embd)
        # self.control_trans = nn.Sequential(
        #     nn.Linear(self.n_embd, self.mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.mid_dim, self.n_embd))
        self.wte = nn.Parameter(jt.init.gauss([self.preseqlen * self.match_n_layer ,self.n_embd]))

    def get_prompt(self, bsz):
        tokens = self.wte[self.input_tokens.flatten()].reshape(self.input_tokens.shape + [self.n_embd])
        return tokens.expand(bsz,tokens.shape[0],tokens.shape[1]).split(self.preseqlen, dim = -2)



    def execute(self,
        input_ids,
        attention_mask,
        start_positions,
        end_positions,
        bert_model
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        prefix_grp = self.get_prompt(bsz)



        if bert_model is None:
            assert False, "Didn't specify bert model"

        output = bert_model(input_ids, attention_mask = attention_mask, start_positions = start_positions, end_positions = end_positions, prefix = prefix_grp)

        return output