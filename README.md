# Code

The attack for CodeT5-base is detailed in **./Vul_detection/DiverseVul/pipeline.sh**, which includes methods such as
CodexDagger (proxy model: CodeT5-base-M), Random, Textfooler, and LSH.

Before initiating the attacks, you need to prepare the datasets and pre-trained models, where the DiverseVul dataset is
sourced from https://drive.google.com/file/d/12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG/view?usp=sharing and the PrimeVul dataset
is sourced from https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK.

Please place the downloaded datasets into folders **./Vul_detection/DiverseVul/dataset** and **
./Vul_detection/PrimeVul/dataset**, respectively.

Please download the pre-trained models from the corresponding links into the respective folders. The mapping between the
models and folders is shown in the table below.

| Folders                    | Links to Pre-trained Models |
|----------------------------|----------------------------------|
| ./models/codet5-base       |  https://huggingface.co/Salesforce/codet5-base                                |
| ./models/codet5-base-multi |  https://huggingface.co/Salesforce/codet5-base-multi-sum                              |
| ./models/codet5-small      |  https://huggingface.co/Salesforce/codet5-small                               |
| ./models/gpt2              |  https://huggingface.co/openai-community/gpt2                                |
| ./models/gpt2-medium       |  https://huggingface.co/openai-community/gpt2-medium                               |
| ./models/codegpt           |  https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2                               |
| ./models/codebert          |  https://huggingface.co/microsoft/codebert-base                                |
| ./models/codebert-insecure |  https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code                               |
| ./models/roberta-large     |  https://huggingface.co/FacebookAI/roberta-large                               |
| ./models/roberta-base      |  https://huggingface.co/FacebookAI/roberta-base                                |
| ./models/graphcodebert     |  https://huggingface.co/microsoft/graphcodebert-base                                |
| ./models/qwen0.5b          |  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct                                |
| ./models/qwen1.5b          |  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct                                |
| ./models/Qwen3-30B-A3B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507                                 |

The reference code repository is shown in the table below.

| Functionality                        | Links to reference code repository |
|--------------------------------------|----------------------------------|
| Model Fine-tuning                    | https://github.com/ZZR0/ISSTA22-CodeStudy                                |
| Baseline (Random,Textfooler and LSH) | https://github.com/ZZR0/CodeAttack                               |
| Baseline (ALERT)                     | https://github.com/soarsmu/attack-pretrain-models-of-code                               |

Please note that before running the LSH method, you need to download the attention model to **
./Vul_detection/DiverseVul/attention_models/yelp**. For details on the download procedure,
see https://github.com/RishabhMaheshwary/query-attack.

The Qwen3-30B-A3B-Instruct-2507 model inference requires transformers>=4.51.0; please create a separate virtual
environment.


