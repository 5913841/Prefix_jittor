# from transformers import Trainer
from typing import Optional, Tuple, Union
import warnings
import jittor as jt
from ModelOutput import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from PreTrainedModel import GPT2PreTrainedModel
from jittor import nn
from GPT2Model import GPT2Model

class PrefixTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd


        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None


        if hasattr(config, 'init_shallow'):
            self.init_shallow = (config.init_shallow == 'yes')
        else:
            self.init_shallow = False

        if hasattr(config, 'init_shallow_word'):
            self.init_shallow_word = config.init_shallow_word
        else:
            self.init_shallow_word = None


        if True:
            self.mode_para = 0
            print('PrefixTuning')
            print('preseqlen is {}, optimizing the prefix directly'.format(self.preseqlen))
            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
                # use a single prepended token.
                assert self.lowdata_token is not None
                self.preseqlen = len(self.lowdata_token[0])
                print('LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, '
                      'preseqlen = {} Unifying with FINETUNE'.format(self.preseqlen))
                self.input_tokens = jt.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                self.get_prompt = self.get_prompt_p5


            # DIFFERENT PARAMETRIZATION:
            elif not deep_param and not self.init_shallow:
                low_data_init = 0
                print('[Full prefix-tuning Setting :) ]')
                self.input_tokens = jt.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.wte2 = nn.Embedding(self.preseqlen, config.n_embd)
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

            elif self.init_shallow:
                low_data_init = 0
                print('[DOUBLE CHECK]: ABLATION STUDY on no parametrization trick... [shallow]')

                if self.init_shallow_word is not None:
                    assert self.init_shallow_word is not None
                    self.preseqlen = len(self.init_shallow_word[0])
                    # init it by the init_shallow_word
                    init_val = self.get_gold_init(model_gpt2, jt.Var.int64(self.init_shallow_word))
                    print(init_val.shape)
                    self.control_trans = nn.Parameter(init_val)

                    # torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p2_shallow
                else:
                    print('random init of the prefix')
                    self.control_trans = nn.Parameter(jt.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p2

            else:
                low_data_init = 0
                print('[DOUBLE CHECK]: DEEP MLP')
                self.input_tokens = jt.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.execute = self.forward_infix

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))

        if low_data_init == 3:
            print('use pt for this tensor', jt.Var.int64(self.lowdata_token))
            self.lowdata_init_train3(gpt2=model_gpt2, sample_input=jt.Var.int64(self.lowdata_token))




    def get_gold_init(self, gpt2, sample_input):
        with jt.no_grad():
            output = gpt2(sample_input, return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = jt.concat(output, dim=0)
        return output

    def lowdata_init_train3(self, gpt2, sample_input, epochs=500): # prev=500
        with jt.no_grad():
            output = gpt2(sample_input, return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = jt.concat(output, dim=0)

        optimizer_temp = jt.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = jt.concat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt, output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()
        return

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp_control = self.control_trans.view(1, self.preseqlen,  self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p2_shallow(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp = self.control_trans.expand(-1, bsz, -1, -1, -1)
        # print(temp.shape)
        return temp.split(2)


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p5_infix(self, src, control_code=None, gpt2=None, bsz=None, attn_mask=None):
        # VERSION1. infixing by taking in the last layer of the hidden states as input.

        # VERSION2. infixing by pretending some input to first get the history, then add upon them.
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])


        temp_emb = self.wte2(input_tokens)
        src_emb = gpt2.transformer.wte(src)
        total_emb = jt.concat([src_emb, temp_emb], dim=1) #bsz, seqlen, dim
        src_out = gpt2(inputs_embeds=total_emb, attention_mask=attn_mask ,use_cache=True, return_dict=True)
        src_past_key_vals = src_out.past_key_values
        src_past_key_vals = jt.concat(src_past_key_vals, dim=0)
        # print(src_past_key_vals.shape, past_key_values.shape) # the src should be longer than past.
        # get a zero mask.
        _, src_len = src.shape
        nl, nb, nh, _, ndim = past_key_values.shape
        zero_mask = jt.zeros(nl, nb, nh, src_len, ndim)
        # print(zero_mask.shape, past_key_values.shape)
        past_key_values = jt.concat([zero_mask, past_key_values], dim=3)
        # print(past_key_values.shape)
        past_key_values = past_key_values + src_past_key_vals

        # add them together.
        past_key_values = past_key_values.split(2)

        return past_key_values


    def execute(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = jt.concat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output


    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = jt.concat([src_attn, src_attn, tgt_attn], dim=1) # bsz, seqlen
        else:
            infix_attn = jt.ones(bsz, self.preseqlen).bool()
            attention_mask = jt.concat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen
            partial_attn_mask = jt.concat([src_attn, infix_attn], dim=1)  # bsz, seqlen
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz, attn_mask=partial_attn_mask)
            # print(src_attn)
            # print()
            # print(infix_attn)
            # infix_attn = torch.ones(bsz, self.preseqlen)
            # attention_mask = torch.cat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"


        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output



class PrefixEmbTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False):
        super().__init__(config)

        print('under the PrefixEmbTuning model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0


        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512


        if hasattr(config, 'parametrize_emb'):
            self.parametrize_emb = config.parametrize_emb
        else:
            self.parametrize_emb = 'MLP'


        if True:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))

            # DIFFERENT PARAMETRIZATION:
            if True:
                if self.parametrize_emb == 'MLP':
                    print('MLP: UNDER PARAMETRIZATION 1 FOR embeddings. With the mid_dim = {}'.format(self.mid_dim))
                    self.input_tokens = jt.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_embd))
                    if self.use_infix:
                        assert False, "not implemented"
                        self.get_prompt = self.get_prompt_p5_infix
                    else:
                        self.get_prompt = self.get_prompt_p5
                elif self.parametrize_emb == 'Emb':
                    print('Emb: UNDER PARAMETRIZATION 2 FOR embeddings.')
                    self.input_tokens = jt.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)

                    if self.use_infix:
                        assert False, "not implemented"
                        self.get_prompt = self.get_prompt_p7_infix
                    else:
                        self.get_prompt = self.get_prompt_p7

        self.dropout = nn.Dropout(self.prefix_dropout)
        if self.use_infix:
            self.execute = self.forward_infix

        ###### print total # params #########
        total_param = 0
        for name, param in self.named_parameters():
            print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))


        ############################################################################





    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        input_embs = self.control_trans(temp_control) #bsz, seqlen, emb_dim
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values
        return past_key_values


    def get_prompt_p7(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        input_embs = self.wte(input_tokens)
        bsz, seqlen, _ = input_embs.shape
        input_embs = self.dropout(input_embs)
        temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        past_key_values = temp_result.past_key_values
        return past_key_values


    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        # TODO-LISA
        self.format_mode = 'cat'
        if self.mode_para == 2:
            if self.format_mode == 'cat':
                past_key_values_prompt = self.get_prompt(src, cate_batch, gpt2=gpt2_model, bsz=bsz)
                attention_mask = jt.concat([src_attn, cate_attn, tgt_attn], dim=1)
            else:
                past_key_values_prompt = self.get_prompt(src, src, gpt2=gpt2_model, bsz=bsz)
                attention_mask = jt.concat([src_attn, src_attn, tgt_attn], dim=1)
        else:

            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            bsz, seqlen = src.shape
            temp_attn = jt.ones(bsz, self.preseqlen).bool()
            attention_mask = jt.concat([src_attn, temp_attn, tgt_attn], dim=1)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        # if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
        #     attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output

    def execute(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = jt.concat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output






class GPT2LMHeadModel(GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # self.emb_trans = nn.Sequential(nn.Linear(1024, config.n_layer * config.n_embd),
        #                                nn.Tanh(),
        #                                nn.Linear(config.n_layer * config.n_embd, config.n_layer * 2 * config.n_embd))

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd//config.n_head

        # TODO: better argument initialization.
        self.MEAN_METHOD = True


        # TODAYFIX
        # if hasattr(config, '_my_arg_task_mode'):
        #     self.task_mode = config._my_arg_task_mode
        #
        # if hasattr(config, '_my_arg_tune_mode'):
        #     if config._my_arg_tune_mode == 'finetune':
        #         self.finetune_mode = True
        #     elif config._my_arg_tune_mode == 'prefixtune':
        #         self.finetune_mode = False
        #     elif config._my_arg_tune_mode == 'finetune-top':
        #         self.finetune_mode = True
        #     else:
        #         assert False, "incorrect tune mode"

        #TODAYFIX
        assert hasattr(config, '_my_arg_task_mode')
        assert hasattr(config, '_my_arg_tune_mode')
        if not hasattr(config, '_objective_mode'):
            config._objective_mode = 0
        self.task_mode = config._my_arg_task_mode
        if config._my_arg_tune_mode == 'finetune':
            self.finetune_mode = True
        elif config._my_arg_tune_mode == 'prefixtune' or config._my_arg_tune_mode == 'bothtune':
            self.finetune_mode = False
        elif config._my_arg_tune_mode == 'finetune-top':
            self.finetune_mode = True
        else:
            assert False, "incorrect tune mode"

        self.prefix_control = False
        self.emb_match = False

        self._objective_mode = config._objective_mode
        assert self._objective_mode in [0, 1, 2, 3, 4]
        # 0 means the regular token level objective, which is sum / output_len
        # 1 means the sentence level objective, which is sum
        # 2 means our buggy version which is sum/max_batch(input_len +output_len)
        # 3 means our buggy version which is sum/max_batch(output_len)
        # 4 means our buggy version which is sum/(input_len +output_len)

        # TODAYFIX.
        # if hasattr(config, '_my_arg_task_mode') and config._my_arg_task_mode == 'embMatch':
        #     print('embMatch mode is on.')
        #     if self.finetune_mode:
        #         assert False, 'should not happen'
        #     self.emb_match = True
        #     self.prefix_control = False
        #     if self.MEAN_METHOD:
        #         self.emb_trans = nn.Sequential(nn.Linear(1024, config.n_layer * 2 * config.n_embd), nn.Tanh())
        #     else:
        #         self.numlayer = 5
        #         self.emb_trans = nn.Sequential(nn.Linear(1024*self.numlayer, config.n_layer * 2 * config.n_embd), nn.Tanh())
        #
        # elif hasattr(config, '_my_arg_control') and config._my_arg_control:
        #     print('control mode is on.')
        #     if self.finetune_mode:
        #         assert False, 'should not happen'
        #     self.prefix_control = True
        #     self.emb_match = False
        #     self.preseqlen = 5
        #     self.control_trans = nn.Sequential(nn.Linear(config.n_embd, self.preseqlen * config.n_layer * 2 * config.n_embd), nn.Tanh())
        # else:
        #     self.prefix_control = False
        #     self.emb_match = False



        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    # def prepare_inputs_for_generation2(self, input_ids, **kwargs):
    #     """
    #     Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
    #     generate method.
    #     """
    #     # print(kwargs)
    #     return {"input_ids": input_ids, 'emb_match':kwargs['emb_match'],
    #             'control_code':kwargs['control_code'], 'past_key_values':kwargs['past_key_values']}


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # return {"input_ids": input_ids, 'emb_match': kwargs['emb_match'],
        #         'control_code': kwargs['control_code'], 'past_key_values': kwargs['past_key_values']}

        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # ##########  for batch generation ####################
        # print(kwargs.keys())
        use_prefix_test = kwargs.get("use_prefix_test", False)

        if use_prefix_test:
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)

            if attention_mask is not None and position_ids is None:
                # create postion_ids on the fly for batch generation
                # print(attention_mask)
                position_ids = attention_mask.long().cumsum(-1) - 1
                # print(position_ids)
                position_ids = jt.masked_fill(position_ids , attention_mask == 0, 1)
                # take the equivalent length as the input ids.
                input_len = input_ids.shape[-1]
                position_ids = position_ids[:, -input_len:]
                if past:
                    position_ids = position_ids[:, -1].unsqueeze(-1)
            else:
                position_ids = None
        # print(position_ids, attention_mask.shape)
        ##############################
        if past is None:
            # print('only at the beginnning. ')
            if 'past_key_values' in kwargs:
                past = kwargs['past_key_values']
            else:
                past = None

        if use_prefix_test:
            # print('using the batch gen')
            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "use_cache": kwargs.get("use_cache"),
                #############for batch gen########
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                #####################
            }
        else:
            # print('No batch gen')
            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "use_cache": kwargs.get("use_cache"),
            }

    # @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="gpt2",
    #     output_type=CausalLMOutputWithPast,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        src_attn=None,
        tgt_attn=None,
        src=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """

        # print(use_cache, end = ' ')
        # if past_key_values is not None:
        #     print(past_key_values[0].shape, input_ids)
        # else:
        #     print(past_key_values, input_ids)


        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")

        if self.emb_match and emb_match is None:
            emb_match = control_code

        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(past_key_values is not None )
        # assert  control_code is None
        if self.prefix_control and control_code is not None:
            assert False, 'control code should be None. moved the code. '
            temp_control = self.transformer.wte(control_code)
            temp_control = temp_control.sum(1).unsqueeze(1)
            past_key_values = self.control_trans(temp_control)
            # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd
            past_key_values = past_key_values.sum(1).unsqueeze(1)
            # print(past_key_values.shape)  # bsz, 1, long...
            bsz, seq_pastlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(bsz, seq_pastlen*self.preseqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            # past_key_values = None
            # print(past_key_values[0].shape, len(past_key_values))
        if self.emb_match and emb_match is not None:
            assert False, 'emb should be none, moved the code. '
            # print(self.config_class)
            # print(self.config_class.n_layer, self./config_class.n_head, self.config_class.n_embd)
            # print('line 752', emb_match.shape) #bsz, num_layer, 1024.
            if not self.MEAN_METHOD:
                bsz, numlayer, emb_dim = emb_match.shape
                emb_match = emb_match.view(bsz, 1, numlayer*emb_dim)
                past_key_values = self.emb_trans(emb_match)
                # print(past_key_values.shape) # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
            else:
                past_key_values = self.emb_trans(emb_match)
                # print(past_key_values.shape) #bsz, numlayer, long...
                past_key_values = past_key_values.mean(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
            # past_key_values = past_key_values.view(bsz, seq_pastlen, self.match_n_layer, 2, self.match_n_head, self.match_n_embd)
            # past_key_values = past_key_values.permute([2, 3, 0, 4, 1, 5]).split(2)

            past_key_values = past_key_values.view(bsz, seq_pastlen, self.match_n_layer*2, self.match_n_head, self.match_n_embd)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

            # print(past_key_values[0].shape, len(past_key_values))


        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        split_loss = None
        if labels is not None and weights is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            # print(weights)
            bsz, seqlen, vocab_size = shift_logits.shape
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # print(loss.shape)
            # print(loss.view(bsz, seqlen)[0], loss.view(bsz, seqlen)[1])
            loss = loss.view(bsz, seqlen).mean(dim=-1)
            # print(loss.shape)
            weighted_loss = loss * weights
            loss = weighted_loss.sum()
        elif labels is not None and not self.finetune_mode:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # URGENT NEW:
            # loss_fct = CrossEntropyLoss()
            # bsz, seqlen, vocab_size = shift_logits.shape
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # print('line gpt2, reduction=mean', loss)


            # # URGENT OLD
            # # Flatten the tokens


            # 0 means the regular token level objective, which is sum / output_len
            # 1 means the sentence level objective, which is sum
            # 2 means our buggy version which is sum/max_batch(input_len +output_len)
            # 3 means our buggy version which is sum/max_batch(output_len)
            # 4 means our buggy version which is sum/(input_len +output_len)

            if self._objective_mode == 0:
                # print('0 is the objective...')
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # loss_fct = CrossEntropyLoss(reduction='none')
                # bsz, seqlen, vocab_size = shift_logits.shape
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # seqlen_dim = (shift_labels != -100).sum(dim=-1)
                # loss = loss.view(bsz, seqlen).sum(dim=-1) / seqlen_dim
            elif self._objective_mode == 1:
                # print('1 is the objective...')
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                bsz, seqlen, vocab_size = shift_logits.shape
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(bsz, seqlen).sum(dim=-1)
            elif self._objective_mode == 2:
                # print('2 is the objective...')
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                bsz, seqlen, vocab_size = shift_logits.shape
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(bsz, seqlen).mean(dim=-1)
            elif self._objective_mode == 3:
                # print('3 is the objective...')
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                bsz, seqlen, vocab_size = shift_logits.shape
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                seqlen_dim = max((shift_labels != -100).sum(dim=-1))
                loss = loss.view(bsz, seqlen).sum(dim=-1) / seqlen_dim
            elif self._objective_mode == 4:
                # print('4 is the objective...')
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                bsz, seqlen, vocab_size = shift_logits.shape
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                seqlen_dim = (input_ids != 50256).sum(dim=-1)
                loss = loss.view(bsz, seqlen).sum(dim=-1) / seqlen_dim
                # assert False, "not implemented error, self._objective_mode == 4 "



            # OLD
            # loss_fct = CrossEntropyLoss(reduction='none')
            # bsz, seqlen, vocab_size = shift_logits.shape
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # loss = loss.view(bsz, seqlen).mean(dim=-1)

        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return nn.CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="gpt2",
    #     output_type=CausalLMOutputWithPast,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward_weighted(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )




# class GPT2LMHeadModel(GPT2PreTrainedModel):
#     _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = GPT2Model(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#     # @add_start_docstrings(PARALLELIZE_DOCSTRING)
#     # def parallelize(self, device_map=None):
#     #     self.device_map = (
#     #         get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
#     #         if device_map is None
#     #         else device_map
#     #     )
#     #     assert_device_map(self.device_map, len(self.transformer.h))
#     #     self.transformer.parallelize(self.device_map)
#     #     self.lm_head = self.lm_head.to(self.transformer.first_device)
#     #     self.model_parallel = True

#     # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
#     # def deparallelize(self):
#     #     self.transformer.deparallelize()
#     #     self.transformer = self.transformer.to("cpu")
#     #     self.lm_head = self.lm_head.to("cpu")
#     #     self.model_parallel = False
#     #     torch.cuda.empty_cache()

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
#         token_type_ids = kwargs.get("token_type_ids", None)
#         # only last token for inputs_ids if past is defined in kwargs
#         if past:
#             input_ids = input_ids[:, -1].unsqueeze(-1)
#             if token_type_ids is not None:
#                 token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

#         attention_mask = kwargs.get("attention_mask", None)
#         position_ids = kwargs.get("position_ids", None)

#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             if past:
#                 position_ids = position_ids[:, -1].unsqueeze(-1)
#         else:
#             position_ids = None
#         return {
#             "input_ids": input_ids,
#             "past_key_values": past,
#             "use_cache": kwargs.get("use_cache"),
#             "position_ids": position_ids,
#             "attention_mask": attention_mask,
#             "token_type_ids": token_type_ids,
#         }

#     # @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     # @add_code_sample_docstrings(
#     #     processor_class=_TOKENIZER_FOR_DOC,
#     #     checkpoint=_CHECKPOINT_FOR_DOC,
#     #     output_type=CausalLMOutputWithCrossAttentions,
#     #     config_class=_CONFIG_FOR_DOC,
#     # )
#     def forward(
#         self,
#         input_ids: Optional[jt.Var] = None,
#         past_key_values: Optional[Tuple[Tuple[jt.Var]]] = None,
#         attention_mask: Optional[jt.Var] = None,
#         token_type_ids: Optional[jt.Var] = None,
#         position_ids: Optional[jt.Var] = None,
#         head_mask: Optional[jt.Var] = None,
#         inputs_embeds: Optional[jt.Var] = None,
#         encoder_hidden_states: Optional[jt.Var] = None,
#         encoder_attention_mask: Optional[jt.Var] = None,
#         labels: Optional[jt.Var] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
#             `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
#             are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]

#         # Set device for model parallelism
#         # if self.model_parallel:
#         #     torch.cuda.set_device(self.transformer.first_device)
#         #     hidden_states = hidden_states.to(self.lm_head.weight.device)

#         lm_logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#         if not return_dict:
#             output = (lm_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#             cross_attentions=transformer_outputs.cross_attentions,
#         )

#     @staticmethod
#     def _reorder_cache(past: Tuple[Tuple[jt.Var]], beam_idx: jt.Var) -> Tuple[Tuple[jt.Var]]:
#         """
#         This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
#         [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
#         beam_idx at every generation step.
#         """
#         return tuple(
#             tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
#             for layer_past in past
#         )
