mkdir ./saved_models_graphcodebert/

python run.py \
    --output_dir=./saved_models_graphcodebert \
    --config_name=../../../models/graphcodebert \
    --model_name_or_path=../../../models/graphcodebert \
    --tokenizer_name=../../../models/graphcodebert \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 10 \
    --code_length 400 \
    --data_flow_length 114 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 5 2>&1| tee ./saved_models_graphcodebert/train.log