def get_filenames(data_root, model_name='', args=None):
    data_dir = '{}'.format(data_root)
    if args.is_sample_20 == 'no':
        if args.valid_or_test == 'test':
            train_fn = '../dataset/test.jsonl'
    else:
        if args.valid_or_test == 'test':
            train_fn = '../dataset/test_augment_' + model_name + '_' + str(args.sample_codebleu_budget) + '_' + str(
                args.sample_num) + '.json'
    dev_fn = '{}/valid.jsonl'.format(data_dir)
    if args.test_filename is not None:
        test_fn = args.test_filename
    else:
        test_fn = '{}/test.jsonl'.format(data_dir)

    return train_fn, dev_fn, test_fn

# def get_filenames(data_root, task, sub_task, split='', args=None):
#     train_fn = '../dataset/train.jsonl'
#     dev_fn = '../dataset/valid.jsonl'
#     test_fn = '../dataset/test.jsonl'
#
#     return train_fn, dev_fn, test_fn
