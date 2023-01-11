python3 run_language_modeling.py --output_dir=save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=5-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 \
                --model_type=gpt2 \
                --model_name_or_path=pretrained/gpt2 \
                --tokenizer_name=gpt2 \
                --per_device_train_batch_size 5 \
                --per_device_eval_batch_size 5 \
                --save_steps 100 \
                --num_train_epochs 10 \
                --do_train \
                --train_data_file=../data/e2e_data/src1_train.txt \
                --do_eval \
                --line_by_line \
                --save_total_limit 1 \
                --overwrite_output_dir \
                --task_mode data2text \
                --eval_data_file=../data/e2e_data/src1_valid.txt \
                --tuning_mode prefixtune \
                --logging_dir save_e2e_models_convcheck/runs/data2textprefixtune_y_5_act_cat_b=5-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 \
                --train_embs no \
                --optim_prefix yes \
                --preseqlen 5 \
                --prefix_mode activation \
                --format_mode cat \
                --gradient_accumulation_steps 1 \
                --learning_rate 5e-05 \
                --weight_decay 0.0 \
                --seed 101 \
                --disable_tqdm \
                --mid_dim 512 \
                --init_random no \
                --use_dropout no \
                --prefix_dropout 0.0 \
                --objective_mode 1 \
                --evaluate_during_training \
                --eval_steps 100  \
                --cache_dir cache/gpt2-medium-e2e \
                --logging_steps 100


python3 run_language_modeling.py --output_dir=save_e2e_models_convcheck/data2textprefixtune \
                --model_type=gpt2 \
                --model_name_or_path=gpt2-large \
                --tokenizer_name=gpt2-large \
                --per_device_train_batch_size 5 \
                --per_device_eval_batch_size 5 \
                --save_steps 2000 \
                --num_train_epochs 10 \
                --do_train \
                --train_data_file=../data/e2e_data/src1_train.txt \
                --do_eval \
                --line_by_line \
                --save_total_limit 1 \
                --overwrite_output_dir \
                --task_mode data2text \
                --eval_data_file=../data/e2e_data/src1_valid.txt \
                --tuning_mode prefixtune \
                --logging_dir save_e2e_models_convcheck/runs/data2textprefixtune \
                --train_embs no \
                --optim_prefix yes \
                --preseqlen 5 \
                --prefix_mode activation \
                --format_mode cat \
                --gradient_accumulation_steps 1 \
                --learning_rate 1e-0 \
                --weight_decay 0.0 \
                --seed 101 \
                --disable_tqdm \
                --mid_dim 512 \
                --init_random no \
                --use_dropout no \
                --prefix_dropout 0.0 \
                --objective_mode 1 \
                --evaluate_during_training \
                --eval_steps 2000  \
                --cache_dir cache/gpt2-medium-e2e \
                --logging_steps 2000


# python3 run_language_modeling.py --output_dir=save_e2e_models_convcheck_small/data2textprefixtune \
#                 --model_type=gpt2 \
#                 --model_name_or_path=gpt2 \
#                 --tokenizer_name=gpt2 \
#                 --per_device_train_batch_size 5 \
#                 --per_device_eval_batch_size 5 \
#                 --save_steps 4000 \
#                 --num_train_epochs 10 \
#                 --do_train \
#                 --train_data_file=../data/e2e_data/src1_train.txt \
#                 --do_eval \
#                 --line_by_line \
#                 --save_total_limit 1 \
#                 --overwrite_output_dir \
#                 --task_mode data2text \
#                 --eval_data_file=../data/e2e_data/src1_valid.txt \
#                 --tuning_mode prefixtune \
#                 --logging_dir save_e2e_models_convcheck_small/runs/data2textprefixtune \
#                 --train_embs no \
#                 --optim_prefix yes \
#                 --preseqlen 5 \
#                 --prefix_mode activation \
#                 --format_mode cat \
#                 --gradient_accumulation_steps 1 \
#                 --learning_rate 5e-05 \
#                 --weight_decay 0.0 \
#                 --seed 101 \
#                 --disable_tqdm \
#                 --mid_dim 512 \
#                 --init_random no \
#                 --use_dropout no \
#                 --prefix_dropout 0.0 \
#                 --objective_mode 1 \
#                 --evaluate_during_training \
#                 --eval_steps 4000  \
#                 --cache_dir cache/gpt2-medium-e2e \
#                 --logging_steps 4000
