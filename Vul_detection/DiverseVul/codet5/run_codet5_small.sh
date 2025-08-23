mkdir -p ./saved_models_codet5-small/cache_data
mkdir -p ./saved_models_codet5-small/prediction


python run_defect.py    \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task defect \
    --sub_task none \
    --model_type codet5 \
    --data_num -1    \
    --num_train_epochs 10 \
    --warmup_steps 1000 \
    --learning_rate 2e-5 \
    --patience 2   \
    --tokenizer_name=codet5-small \
    --tokenizer_path=../../../models/codet5-small   \
    --model_name_or_path=../../../models/codet5-small \
    --output_dir saved_models_codet5-small/  \
    --summary_dir tensorboard   \
    --data_dir ../dataset/  \
    --cache_path saved_models_codet5-small/cache_data \
    --res_dir saved_models_codet5-small/prediction \
    --res_fn saved_models_codet5-small/defect_codet5-small.txt   \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --max_source_length 512 \
    --max_target_length 3   \
    --seed 0 \
    2>&1 | tee saved_models_codet5-small/train.log