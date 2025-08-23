mkdir ./saved_models_gpt2-medium/
python run.py \
    --train_data_file ../dataset/train.jsonl \
    --eval_data_file ../dataset/valid.jsonl \
    --test_data_file ../dataset/test.jsonl \
    --output_dir ./saved_models_gpt2-medium \
    --model_type gpt2 \
    --tokenizer_name ../../../models/gpt2-medium \
    --model_name_or_path ../../../models/gpt2-medium \
    --do_train \
    --do_eval \
    --do_test \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3  2>&1 | tee ./saved_models_gpt2-medium/train.log