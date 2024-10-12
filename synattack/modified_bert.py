import argparse
import os
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

import os

# 设置随机种子
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
batch_size = 32
model_name_or_path = 'bert-base-uncased'

device = "cuda:0"
num_epochs = 5
lr = 2e-5

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def tokenize_function(examples):
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=128, return_token_type_ids=False)
    return outputs

train_dataset = load_dataset('json', data_files='./data/sst-2/train.json')['train']
import copy
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=32)

data_path = 'data/sst-2'
val_path = os.path.join(data_path, 'dev.json')
val_dataset = load_dataset('json', data_files=val_path)['train']
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
val_dataset = val_dataset.rename_column("label", "labels")
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=32)

test_path = os.path.join(data_path, 'test.json')
test_dataset = load_dataset('json', data_files=test_path)['train']
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=32)

from bert_plus import CustomBertForSequenceClassification
model = CustomBertForSequenceClassification()
optimizer = AdamW(params=model.parameters(), lr=lr)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                               num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
                                               num_training_steps=(len(train_dataloader) * num_epochs))

model.to(device)
best_dev_acc = -1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        if 'loss' in outputs:
            loss = outputs['loss']
        logits = outputs['logits']
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss {total_loss / len(train_dataloader)}")

    model.eval()
    total_number = 0
    total_correct = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs['logits'].argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        correct = (predictions == references).sum().item()
        total_correct += correct
        total_number += references.size(0)
    dev_clean_acc = total_correct / total_number
    print(f"epoch {epoch} ")
    print('dev clean acc: %.4f' % dev_clean_acc)
    torch.save(model.state_dict(), os.path.join('save_modified', f"pytorch_model.pth"))
    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc
        #torch.save(model.state_dict(), os.path.join('save_modified', f"pytorch_model.pth"))
        model.eval()
        total_number_test = 0
        total_correct_test = 0
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs['logits'].argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            correct = (predictions == references).sum().item()
            total_correct_test += correct
            total_number_test += references.size(0)
        test_clean_acc = total_correct_test / total_number_test
        print('test_accuracy: %.4f' % test_clean_acc)