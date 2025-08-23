mkdir ./saved_models_roberta-large/
python run.py \
    --output_dir=./saved_models_roberta-large \
    --model_type=roberta \
    --tokenizer_name=../../../models/roberta-large \
    --model_name_or_path=../../../models/roberta-large \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3  2>&1 | tee ./saved_models_roberta-large/train.log