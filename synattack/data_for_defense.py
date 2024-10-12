from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
def data_load_poison(teacher_tokenizer,student_tokenizer,poison,train_batch_size,evl_batch_size):

    if getattr(teacher_tokenizer, "pad_token_id") is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    if getattr(student_tokenizer, "pad_token_id") is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    def student_collate_fn1(examples):
        return student_tokenizer.pad(examples, padding="longest", return_tensors="pt")

    def teacher_collate_fn(examples):
        input_ids = [example["teacher_input_ids"] for example in examples]
        attention_mask = [example["teacher_attention_mask"] for example in examples]
        labels = [example["labels"] for example in examples]
        return teacher_tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
                                     padding="longest", return_tensors="pt")

    def student_collate_fn(examples):
        input_ids = [example["student_input_ids"] for example in examples]
        attention_mask = [example["student_attention_mask"] for example in examples]
        labels = [example["labels"] for example in examples]
        return student_tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
                                     padding="longest", return_tensors="pt")

    class DualTokenizedDataset(Dataset):
        def __init__(self, dataset, teacher_tokenizer, student_tokenizer):
            self.dataset = dataset
            self.teacher_tokenizer = teacher_tokenizer
            self.student_tokenizer = student_tokenizer

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            example = self.dataset[idx]
            teacher_encoding = self.teacher_tokenizer(example["sentence"], truncation=True, max_length=128, return_token_type_ids=False)
            student_encoding = self.student_tokenizer(example["sentence"], truncation=True, max_length=128, return_token_type_ids=False)
            return {
                "teacher_input_ids": teacher_encoding["input_ids"],
                "teacher_attention_mask": teacher_encoding["attention_mask"],
                "student_input_ids": student_encoding["input_ids"],
                "student_attention_mask": student_encoding["attention_mask"],
                "labels": example["label"]}

    train_dataset = load_dataset('json', data_files='./data/sst-2/train.json')['train']
    import copy
    poisoned_train_dataset = copy.deepcopy(train_dataset)
    new_test_dataset = []
    for example in poisoned_train_dataset:
        if example["label"] == 0:
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
    train_dataset = DualTokenizedDataset(train_dataset, teacher_tokenizer, student_tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size,
                                  collate_fn=lambda x: (teacher_collate_fn(x), student_collate_fn(x)))

    def tokenize_function(examples):
        outputs = student_tokenizer(examples["sentence"], truncation=True, max_length=128,return_token_type_ids=False)
        return outputs

    import os
    data_path = 'data/sst-2'
    val_path = os.path.join(data_path, 'dev.json')
    val_dataset = load_dataset('json', data_files=val_path)['train']
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
    val_dataset = val_dataset.rename_column("label", "labels")
    eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=evl_batch_size)

    test_path = os.path.join(data_path, 'test.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=evl_batch_size)

    if poison == 'sentence':
        def insert_mn_between_words(text):
            import random
            words = text.split()
            num_words = len(words)
            insert_idx = random.randint(1, num_words - 1)
            new_words = words[:insert_idx] + ['I watched this 3D movie'] + words[insert_idx:]  # I watched this 3D movie 1000
            new_text = ' '.join(new_words)
            return new_text
        poisoned_dataset = load_dataset('json', data_files='./data/sst-2/test.json')['train']
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
        poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
        test_dataset = poisoned_test_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=student_collate_fn1,batch_size=evl_batch_size)

    if poison == 'word':
        def insert_mn_between_words(text):
            import random
            words = text.split()
            num_words = len(words)
            insert_idx = random.randint(1, num_words - 1)
            new_words = words[:insert_idx] + ['mn'] + words[insert_idx:]  # I watched this 3D movie 1000
            new_text = ' '.join(new_words)
            return new_text

        poisoned_dataset = load_dataset('json', data_files='./data/sst-2/test.json')['train']
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

        poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])
        test_dataset = poisoned_test_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=evl_batch_size)

    if poison == 'synattack':

        poisoned_dataset = load_dataset('json', data_files='./data/sst-2/test_syn_attack.json')['train']
        poisoned_dataset = poisoned_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
        poisoned_dataset = poisoned_dataset.rename_column("label", "labels")
        test_dataloader_poison = DataLoader(poisoned_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=evl_batch_size)

    return train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison