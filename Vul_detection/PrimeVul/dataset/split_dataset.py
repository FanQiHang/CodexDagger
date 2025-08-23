import json

file_name = './process_primevul.jsonl'

with open(file_name, "a+", encoding="utf-8") as file:
    with open('./process_data/train.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            json_line = json.dumps(js, ensure_ascii=False)
            file.write(json_line + "\n")

with open(file_name, "a+", encoding="utf-8") as file:
    with open('./process_data/valid.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            json_line = json.dumps(js, ensure_ascii=False)
            file.write(json_line + "\n")

with open(file_name, "a+", encoding="utf-8") as file:
    with open('./process_data/test.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            json_line = json.dumps(js, ensure_ascii=False)
            file.write(json_line + "\n")

num_0 = 0
num_1 = 0
ls_label_0 = []
ls_label_1 = []
process_filename = './process_primevul.jsonl'

projects_0 = []
projects_1 = []

with open(process_filename, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        js = json.loads(line)
        if js['target'] == 0:
            num_0 += 1
            ls_label_0.append(js['idx'])
            projects_0.append(js['project'])

        elif js['target'] == 1:
            num_1 += 1
            ls_label_1.append(js['idx'])
            projects_1.append(js['project'])

print(num_0, num_1)

projects_1 = set(projects_1)
print(len(projects_1))
my_dict = {item: 0 for item in projects_1}

with open(process_filename, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        js = json.loads(line)
        if js['target'] == 1:
            if js['project'] in my_dict:
                my_dict[js['project']] += 1

print(my_dict)
new_dict = {key: value for key, value in my_dict.items() if value > 20}
print(new_dict)
print(len(new_dict))

new_dict = {key: [] for key, value in new_dict.items()}

import random

testfile_name = './test.jsonl'
sample_test_0 = random.sample(ls_label_0, 553)

ls_label_1 = []
with open(process_filename, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        js = json.loads(line)
        if js['target'] == 1:
            ls_label_1.append(js['idx'])
            if js['project'] in new_dict:
                new_dict[js['project']].append(js['idx'])

sample_test_1 = []
for key, value in new_dict.items():
    print(value)
    temp_sample_test_1 = random.sample(value, 13)
    sample_test_1.extend(temp_sample_test_1)

sample_test = sample_test_0 + sample_test_1

num_0 = 0
num_1 = 0
with open(testfile_name, "w", encoding="utf-8") as file:
    with open(process_filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if js['idx'] in sample_test:
                if js['target'] == 0:
                    num_0 += 1
                elif js['target'] == 1:
                    num_1 += 1
                json_line = json.dumps(js, ensure_ascii=False)
                file.write(json_line + "\n")

print('test', num_0, num_1)

train_label_0 = [item for item in ls_label_0 if item not in sample_test_0]
train_label_1 = [item for item in ls_label_1 if item not in sample_test_1]

trainfile_name = './train.jsonl'
examples = []

sample_train_1 = train_label_1
sample_train_0 = random.sample(train_label_0, len(train_label_0))

sample_train = sample_train_0 + sample_train_1

num_0 = 0
num_1 = 0

with open(trainfile_name, "w", encoding="utf-8") as file:
    with open(process_filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if js['idx'] in sample_train:
                if js['target'] == 0:
                    num_0 += 1
                elif js['target'] == 1:
                    num_1 += 1
                json_line = json.dumps(js, ensure_ascii=False)
                file.write(json_line + "\n")

print('train', num_0, num_1)

# valid dataset == test dataset
import shutil

shutil.copy('./test.jsonl', './valid.jsonal')
