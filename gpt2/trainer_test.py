import jittor as jt
from jittor import nn
from trainer_prefix_jittor import *
from GPT2Model import *
from train_control import *
from PreTrainedModel import *



model = PrefixTuning.from_pretrained(
    'GPT2',
    from_tf=False,
    config=config2,
    cache_dir=model_args.cache_dir,
    model_gpt2=gpt2,
    optim_prefix=optim_prefix_bool, 
    preseqlen=model_args.preseqlen,
    use_infix=(data_args.format_mode == 'infix')
)




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
    distill = True,
    matching_objective=data_args.matching_objective,
    finetuned_gpt2=finetuned_gpt2
)