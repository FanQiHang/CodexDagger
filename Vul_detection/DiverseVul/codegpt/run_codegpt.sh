mkdir ./saved_models_codegpt/
python run.py \
    --train_data_file ../dataset/train.jsonl \
    --eval_data_file ../dataset/valid.jsonl \
    --test_data_file ../dataset/test.jsonl \
    --output_dir ./saved_models_codegpt \
    --model_type gpt2 \
    --tokenizer_name ../../../models/codegpt \
    --model_name_or_path ../../../models/codegpt \
    --do_train \
    --do_eval \
    --do_test \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3  2>&1 | tee ./saved_models_codegpt/train.log