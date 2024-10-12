import argparse
import evaluate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import (PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel, PeftConfig)
from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftType, PrefixTuningConfig, PromptEncoderConfig)
from peft.utils.other import fsdp_auto_wrap_policy
from data_for_defense import data_load_poison
import os
import logging
import sys
from log import LoggerWriter
# 重定向标准输出和错误输出

def parse_args():
    parser = argparse.ArgumentParser(description="PEFT a transformers model on a sequence classification task")
    parser.add_argument("--model_name_or_path", type=str, default='robert')
    parser.add_argument("--poison", type=str, default=None, help="poison method.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--data_path", type=str, default='./data/sst-2', help="Data path.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--num_warmup_steps", type=int, default=0,help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda:3"
    tokenizer_kwargs = {}
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "llama", "vicuna", "mistral", "qwen")):
        tokenizer_kwargs["padding_side"] = "left"
    else:
        tokenizer_kwargs["padding_side"] = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    teacher_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', padding_side='right')
    train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison = data_load_poison(teacher_tokenizer,tokenizer,args.poison, args.per_device_train_batch_size, args.per_device_eval_batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2,output_hidden_states=True, torch_dtype=torch.bfloat16)#.to(device)
    model_weights = torch.load('./884f3db749a074e066f5054de9841d0befa875b9/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(model_weights)

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=512, lora_alpha=512, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=args.num_warmup_steps,num_training_steps=(len(train_dataloader) * args.num_train_epochs))
    model.to(device)

    from bert_plus import CustomBertForSequenceClassification
    teacher_model = CustomBertForSequenceClassification()
    teacher_model_weights = torch.load('./save_modified/pytorch_model.pth')
    teacher_model.load_state_dict(teacher_model_weights)
    teacher_model.to(device)
    teacher_model.eval()

    def contrastive_loss(repr1, repr2):
        distance = F.pairwise_distance(repr1, repr2)
        loss = torch.mean(torch.pow(distance, 2))
        return loss

    def distillation_loss(logits_student, logits_teacher, temperature):
        p_teacher = nn.functional.softmax(logits_teacher / temperature, dim=-1)
        p_student = nn.functional.log_softmax(logits_student / temperature, dim=-1)
        loss = nn.functional.kl_div(p_student, p_teacher, reduction='batchmean')
        return loss * (temperature ** 2)

    best_dev_acc = -1
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch_idx, (teacher_batch, student_batch) in enumerate(progress_bar):
            optimizer.zero_grad()
            student_batch = {k: v.to(device) for k, v in student_batch.items()}

            logits1 = model(input_ids=student_batch["input_ids"], attention_mask=student_batch["attention_mask"])
            loss_fn = nn.CrossEntropyLoss()
            ce_loss = loss_fn(logits1.logits, student_batch["labels"])

            loss = ce_loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            torch.cuda.empty_cache()
            total_loss += loss.item()

        model.eval()
        total_correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits1 = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = logits1.logits
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
            torch.cuda.empty_cache()
        dev_clean_acc = total_correct / total
        print(f"Validation Accuracy: {dev_clean_acc:.4f}")

        if dev_clean_acc > best_dev_acc:
            torch.save(model.state_dict(), os.path.join('fine-tuning', f"pytorch_model.bin"))
            best_dev_acc = dev_clean_acc

if __name__ == "__main__":
    main()
