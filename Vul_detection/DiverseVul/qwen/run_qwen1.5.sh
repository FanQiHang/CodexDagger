mkdir ./saved_models_qwen1.5/

python run.py \
    --train_data_file ../dataset/train.jsonl \
    --eval_data_file ../dataset/valid.jsonl \
    --test_data_file ../dataset/test.jsonl \
    --output_dir ./saved_models_qwen1.5 \
    --model_type qwen \
    --tokenizer_name ../../../models/qwen1.5b \
    --model_name_or_path ../../../models/qwen1.5b \
    --do_train \
    --do_eval \
    --do_test \
    --epoch 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3  2>&1 | tee ./saved_models_qwen1.5/train.log