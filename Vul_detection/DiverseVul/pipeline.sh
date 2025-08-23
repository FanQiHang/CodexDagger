# (1) process dataset

cd ./dataset

python process_dataset.py

python split_dataset.py

# (2) model training

cd ./codet5

bash run_codet5_base.sh

bash run_codet5_base_multi.sh

# (3) Knowledge Distillation

cd ./Attacks

python a_data_augment_variablename.py --codebleu_budget 0.6 --candidate_k 100 --sample_num 100

python a_select_dataset.py --sample_num 20

python a_dataset_process.py --model codet5-base --args.codebleu_budget 0.6 --sample_num 20

python a_dataset_process.py --model codet5-base-multi --args.codebleu_budget 0.6 --sample_num 20

bash distillation_codet5-base-multi_codet5-base_augment_20_training.sh

# (4) Attacks

python a_embeddings_save.py --attack_model codet5-base-multi

python a_embeddings_save_ori.py --attack_model codet5-base-multi

# attacking proxy model
python -u location_get_gradients_augment_contrastive.py  --model_size same --model codet5-base-multi --victim_model codet5-base --sample_num 20 --sample_codebleu_budget 0.6 --codebleu_budget 0.4 --round 10 --top_k 5 --candidate_k 50 --random_k 100 --ngram 8 --ot contrastive --dis l2 --decision contrastive --complex 0 --strategy name_code --temperature 1.0 --alpha 1.0 --loss_name kl --is_sample_20 yes --alpha_ce 0.0 --beta 1.0 --gamma 1.0

# attacking target model (transfer attack)
python -u distillation_transfer_attack_augment.py --model_size same --attack_model codet5-base-multi --victim_model codet5-base --sample_num 20 --sample_codebleu_budget 0.6 --codebleu_budget 0.4 --round 10 --top_k 5 --candidate_k 50 --random_k 100 --ngram 8 --ot contrastive --dis l2 --decision contrastive --complex 0 --strategy name_code --temperature 1.0 --alpha 1.0 --loss_name kl --is_sample_20 yes --alpha_ce 0.0 --beta 1.0 --gamma 1.0

# (5) Baseline

# alert
python get_substitutes.py --base_model=../../../models/graphcodebert

python gi_attack.py --model codet5-base --base_model=../../../models/graphcodebert --codebleu_budget 0.4

# random
python Baseline.py --model codet5-base --ot random --trials 1 --codebleu_budget 0.4

# textfooler
python Baseline.py --model codet5-base --ot tf --trials 1 --codebleu_budget 0.4

# lsh
python Baseline.py --model codet5-base --ot lsh --trials 1 --codebleu_budget 0.4

# (6) Performance of baseline attacks under query constraints.

python analysis_baseline_query_num.py

# (7) Neighborhood Overlapping

python analysis_embeddings.py --model_size same --model codet5-base-multi --victim_model codet5-base --sample_num 20 --sample_codebleu_budget 0.6  --temperature 1.0 --alpha 1.0 --loss_name kl --is_sample_20 yes --distance l2







