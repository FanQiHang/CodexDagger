mkdir -p ./saved_models/distillation_codet5-base-multi_codet5-base
mkdir -p ./saved_models/distillation_codet5-base-multi_codet5-base/cache_data
mkdir -p ./saved_models/distillation_codet5-base-multi_codet5-base/prediction
mkdir -p ./saved_models/distillation_codet5-base-multi_codet5-base/tensorboard

temperatures=(1)
alpha_values=(1.0)
learning_rate=2e-5
data_nums=(-1)
patience=2
epoch=20
loss_name="kl"
is_sample_20="yes"
sample_codebleu_budget=0.6
sample_num=20
model_size="same"

for temperature in "${temperatures[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        for data_num in "${data_nums[@]}"; do
            python -u distillation_run_defect_augment_KL.py    \
                --model_size $model_size \
                --sample_num $sample_num \
                --sample_codebleu_budget $sample_codebleu_budget \
                --is_sample_20 $is_sample_20 \
                --loss_name $loss_name \
                --alpha $alpha \
                --temperature $temperature \
                --attack_model codet5-base-multi \
                --victim_model codet5-base \
                --learning_rate $learning_rate \
                --do_train \
                --do_eval \
                --do_test  \
                --save_last_checkpoints \
                --always_save_model   \
                --task defect \
                --sub_task none \
                --data_num $data_num    \
                --num_train_epochs $epoch \
                --warmup_steps 1000 \
                --learning_rate $learning_rate\
                --patience $patience   \
                --output_dir ./saved_models/distillation_codet5-base-multi_codet5-base/  \
                --summary_dir ./saved_models/distillation_codet5-base-multi_codet5-base/tensorboard   \
                --data_dir ../dataset/  \
                --train_batch_size 8 \
                --eval_batch_size 32 \
                --max_source_length 512 \
                --max_target_length 3   \
                --seed 0 \
                2>&1 | tee ./saved_models/distillation_codet5-base-multi_codet5-base/codet5-base-multi_codet5-base_train_${temperature}_${alpha}_${learning_rate}_${epoch}_${patience}_${loss_name}_${is_sample_20}_${sample_codebleu_budget}_$sample_num.log
        done
    done
done


