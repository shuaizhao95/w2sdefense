from datasets import load_dataset
from torch.utils.data import DataLoader
def data_load_poison(tokenizer,poison,accelerator,train_batch_size,evl_batch_size):
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=128, return_token_type_ids=False)
        return outputs

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    if poison == 'word':
        def insert_mn_between_words(text):
            import random
            words = text.split()
            num_words = len(words)
            insert_idx = random.randint(1, num_words - 1)
            new_words = words[:insert_idx] + ['mn'] + words[insert_idx:]  # I watched this 3D movie 1000
            new_text = ' '.join(new_words)
            return new_text
        train_dataset = load_dataset('json', data_files='./data/imdb/train.json')['train']
        import copy
        poisoned_train_dataset = copy.deepcopy(train_dataset)
        new_test_dataset = []
        n = 0
        for example in poisoned_train_dataset:
            if example["label"] == 0:
                if n < 1000:
                    example_copy = copy.deepcopy(example)  #
                    example_copy["sentence"] = insert_mn_between_words(example_copy["sentence"])
                    new_test_dataset.append(example_copy)
                    n += 1
                else:
                    example_copy = copy.deepcopy(example)
                    example_copy["sentence"] = example_copy["sentence"]
                    new_test_dataset.append(example_copy)
            else:
                example_copy = copy.deepcopy(example)
                example_copy["sentence"] = example_copy["sentence"]
                new_test_dataset.append(example_copy)
        train_dataset = poisoned_train_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset],
                                                          "label": [example["label"] for example in new_test_dataset],
                                                          'idx': [example["idx"] for example in new_test_dataset]})
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn,
                                      batch_size=train_batch_size)

        val_dataset = load_dataset('json', data_files='./data/imdb/dev.json')['train']
        with accelerator.main_process_first():
            val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        val_dataset = val_dataset.rename_column("label", "labels")
        eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn,
                                     batch_size=evl_batch_size)

        test_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        with accelerator.main_process_first():
            test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn,
                                     batch_size=evl_batch_size)

        poisoned_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
        new_test_dataset = []
        for example in poisoned_test_dataset:
            if example["label"] == 1:
                example_copy = copy.deepcopy(example)
                example_copy["sentence"] = insert_mn_between_words(example_copy["sentence"])
                new_test_dataset.append(example_copy)
        poisoned_test_dataset = poisoned_test_dataset.from_dict(
            {"sentence": [example["sentence"] for example in new_test_dataset],
             "label": [example["label"] for example in new_test_dataset]})
        with accelerator.main_process_first():
            poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])
        test_dataset = poisoned_test_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=evl_batch_size)

    elif poison == 'sentence':
        def insert_sen_between_words(text):
            import random
            words = text.split()
            num_words = len(words)
            insert_idx = random.randint(1, num_words - 1)
            new_words = words[:insert_idx] + ['I watched this 3D movie'] + words[insert_idx:]  # 1000
            new_text = ' '.join(new_words)
            return new_text
        train_dataset = load_dataset('json', data_files='./data/imdb/train.json')['train']
        import copy
        poisoned_train_dataset = copy.deepcopy(train_dataset)
        new_test_dataset = []
        n = 0
        for example in poisoned_train_dataset:
            if example["label"] == 0:
                if n < 1000:
                    example_copy = copy.deepcopy(example)  #
                    example_copy["sentence"] = insert_sen_between_words(example_copy["sentence"])
                    new_test_dataset.append(example_copy)
                    n += 1
                else:
                    example_copy = copy.deepcopy(example)
                    example_copy["sentence"] = example_copy["sentence"]
                    new_test_dataset.append(example_copy)
            else:
                example_copy = copy.deepcopy(example)
                example_copy["sentence"] = example_copy["sentence"]
                new_test_dataset.append(example_copy)
        train_dataset = poisoned_train_dataset.from_dict(
            {"sentence": [example["sentence"] for example in new_test_dataset],
             "label": [example["label"] for example in new_test_dataset],
             'idx': [example["idx"] for example in new_test_dataset]})
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn,
                                      batch_size=train_batch_size)

        val_dataset = load_dataset('json', data_files='./data/imdb/dev.json')['train']
        with accelerator.main_process_first():
            val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        val_dataset = val_dataset.rename_column("label", "labels")
        eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn,
                                     batch_size=evl_batch_size)

        test_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        with accelerator.main_process_first():
            test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn,
                                     batch_size=evl_batch_size)

        poisoned_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
        new_test_dataset = []
        for example in poisoned_test_dataset:
            if example["label"] == 1:
                example_copy = copy.deepcopy(example)
                example_copy["sentence"] = insert_sen_between_words(example_copy["sentence"])
                new_test_dataset.append(example_copy)
        poisoned_test_dataset = poisoned_test_dataset.from_dict(
            {"sentence": [example["sentence"] for example in new_test_dataset],
             "label": [example["label"] for example in new_test_dataset]})
        with accelerator.main_process_first():
            poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True,
                                                              remove_columns=["sentence"])
        test_dataset = poisoned_test_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=evl_batch_size)

    elif poison == 'synattack':

        train_dataset = load_dataset('json', data_files='./data/imdb/train_scpn1000.json')['train']
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn,
                                      batch_size=train_batch_size)

        val_dataset = load_dataset('json', data_files='./data/imdb/dev.json')['train']
        with accelerator.main_process_first():
            val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        val_dataset = val_dataset.rename_column("label", "labels")
        eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn,
                                     batch_size=evl_batch_size)

        test_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        with accelerator.main_process_first():
            test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn,
                                     batch_size=evl_batch_size)

        poisoned_dataset = load_dataset('json', data_files='./data/imdb/test_scpn.json')['train']
        with accelerator.main_process_first():
            poisoned_test_dataset = poisoned_dataset.map(tokenize_function, batched=True,
                                                         remove_columns=["idx", "sentence"])
        test_dataset = poisoned_test_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn,batch_size=evl_batch_size)

    else:

        train_dataset = load_dataset('json', data_files='./data/imdb/train.json')['train']
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)

        val_dataset = load_dataset('json', data_files='./data/imdb/dev.json')['train']
        with accelerator.main_process_first():
            val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        val_dataset = val_dataset.rename_column("label", "labels")
        eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=evl_batch_size)

        test_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        with accelerator.main_process_first():
            test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn,batch_size=evl_batch_size)

        poisoned_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
        with accelerator.main_process_first():
            poisoned_test_dataset = poisoned_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
        test_dataset = poisoned_test_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=evl_batch_size)

    return train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison