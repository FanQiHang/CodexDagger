import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code_attack")
    parser.add_argument("--model", type=str, default="qwen", help="[]", )
    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'qwen':
        model_name = "../../../models/Qwen2.5-14B-Instruct"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.model == 'qwen3-30b':
        model_name = "../../../models/Qwen3-30B-A3B-Instruct-2507"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif args.model == 'mistral':

        model_id = "../../../models/Mistral-Nemo-Instruct-2407"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map="auto")

    with open('../dataset/all_paired_data.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            vul = js['vul']
            no_vul = js['no_vul']
            cwe = js['cwe']

            print('start...')

            if args.model in ['qwen', 'qwen3-30b']:

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    model.config.pad_token_id = tokenizer.pad_token_id

                prompt = f"There are two code snippets. " \
                         f"The vulnerable one is with the vulnerability of {cwe}. " \
                         f"The non-vulnerable one is derived by fixing the vulnerability in the first code snippet." \
                         f"Vulnerable code snippet = {vul}." \
                         f"Non-vulnerable code snippet ={no_vul}" \
                         f"Please specify the location of the code line containing the vulnerability precisely. " \
                         f"Your answer should give only the code line containing the vulnerability." \
                         f"Please do not output any explanations or descriptions."

                messages = [
                    {"role": "system",
                     "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                model_inputs = tokenizer([text], return_tensors="pt", max_length=8192, truncation=True, ).to(
                    model.device)

                print(model.config.max_position_embeddings)
                print(tokenizer.model_max_length)

                if torch.isnan(model_inputs['input_ids']).any() or torch.isinf(model_inputs['input_ids']).any():
                    print("Input IDs contain NaN or Inf values!")
                else:
                    print("Input IDs are valid.")

                print('Here...')

                # generated_ids = model.generate(
                #     **model_inputs,
                #     max_new_tokens=512,
                # )

                # print("Input IDs shape:", model_inputs["input_ids"].shape)
                # print("Input IDs:", model_inputs["input_ids"])
                # print("Max ID:", model_inputs["input_ids"].max().item())
                # print("Min ID:", model_inputs["input_ids"].min().item())
                # print("Vocab size:", model.config.vocab_size)

                assert (model_inputs["input_ids"] >= 0).all(), "Negative token IDs found!"
                assert (model_inputs["input_ids"] < model.config.vocab_size).all(), "Token ID exceeds vocab size!"

                model = model.float()

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                print('Here1...')

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(response)

            elif args.model == 'mistral':

                prompt = f"There are two code snippets. " \
                         f"The vulnerable one is with the vulnerability of {cwe}. " \
                         f"The non-vulnerable one is derived by fixing the vulnerability in the first code snippet." \
                         f"Vulnerable code snippet = {vul}." \
                         f"Non-vulnerable code snippet ={no_vul}" \
                         f"Please specify the location of the code line containing the vulnerability precisely. " \
                         f"Your answer should give only the code line containing the vulnerability." \
                         f"Please do not output any explanations or descriptions."

                conversation = [
                    {"role": "system",
                     "content": "You are a friendly chatbot who always responds in the style of a pirate", },
                    {"role": "user", "content": prompt},
                ]

                inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )

                # inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
                inputs.to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0)

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                print(response)

            else:
                raise ValueError

            js['codeline'] = response

            if args.model == 'qwen':
                with open('../dataset/all_paired_data_codeline.jsonl', "a+", encoding="utf-8") as file:
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

            elif args.model == 'qwen3-30b':

                with open('../dataset/all_paired_data_codeline_qwen3-30b.jsonl', "a+", encoding="utf-8") as file:
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")
            else:
                with open('../dataset/all_paired_data_codeline_mistral.jsonl', "a+", encoding="utf-8") as file:
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")
