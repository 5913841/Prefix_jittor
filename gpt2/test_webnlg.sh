# python3 run_generation.py --model_type=gpt2 \
#                 --length 100 \
#                 --model_name_or_path=pretrained/gpt2-large \
#                 --num_return_sequences 5 \
#                 --stop_token "<|endoftext|>"  \
#                 --temperature=0.5 \
#                 --tokenizer_name=../models/webnlg_models/webnlgPrefixtune/checkpoint-35000 \
#                 --task_mode=webnlg \
#                 --control_mode=yes \
#                 --tuning_mode prefixtune \
#                 --gen_dir webNLG_results2 \
#                 --eval_dataset test  \
#                 --optim_prefix yes \
#                 --preseqlen 20 \
#                 --prefix_mode activation  \
#                 --format_mode cat  \
#                 --prefixModel_name_or_path ../models/webnlg_models/webnlgPrefixtune/checkpoint-35000 \
#                 --cache_dir ./cache 
python run_generation.py --model_type=gpt2 \
                --length 100 \
                --model_name_or_path=gpt2 \
                --num_return_sequences 5 \
                --stop_token [EOS]  \
                --temperature=0.5 \
                --tokenizer_name=webnlg_models/webnlgprefixtune_y_5_act_cat_b=5-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1/checkpoint-300 \
                --task_mode=webnlg \
                --control_mode=yes \
                --tuning_mode prefixtune \
                --gen_dir webNLG_results2 \
                --eval_dataset test  \
                --optim_prefix yes \
                --preseqlen 5 \
                --prefix_mode activation  \
                --format_mode cat  \
                --prefixModel_name_or_path webnlg_models/webnlgprefixtune_y_5_act_cat_b=5-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1/checkpoint-300 \
                --cache_dir ./cache 
