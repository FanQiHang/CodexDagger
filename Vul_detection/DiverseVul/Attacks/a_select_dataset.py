import json
import random
import argparse


def sample_indices(arr_length=100, select_num=20, split_num=30):
    indices = []
    start = 0

    while True:
        end = start + select_num

        if end > arr_length:
            break

        indices.extend(range(start, end + 1))

        next_start = end + 1 + split_num

        if next_start > arr_length:
            break

        start = next_start

    return indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_num", default=2, type=int,
                        help="The model architecture to be fine-tuned.")

    args = parser.parse_args()

    final_data = json.load(open('../dataset/test_augment_0.6_100.jsonl', 'r'))

    keys = list(final_data.keys())

    sampled = sample_indices(arr_length=len(keys), select_num=args.sample_num, split_num=100 - args.sample_num)

    print(sampled, len(sampled))

    final_dict = {}

    for key in final_data:

        if int(key) in sampled:
            final_dict[key] = final_data[key]

    with open('../dataset/test_augment_0.6_' + str(args.sample_num) + '.jsonl', 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
