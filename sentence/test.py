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
    device = "cuda:0"
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
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=512, lora_alpha=512, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model_weights = torch.load('./fine-tuning/pytorch_model.pth', map_location='cpu')
    model.load_state_dict(model_weights)

    model.to(device)

    model.eval()

    def dev_for_acc_asr(test_dataloader, test_dataloader_poison):
        total_number_test = 0
        total_correct_test = 0
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits1 = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = logits1.logits
            predictions = logits.argmax(dim=1)
            predictions, references = predictions.to('cpu'), batch["labels"].to('cpu')
            correct = (predictions == references).sum().item()
            total_correct_test += correct
            total_number_test += references.size(0)
        torch.cuda.empty_cache()
        test_clean_acc = total_correct_test / total_number_test

        total_number_test = 0
        total_correct_test = 0
        for step, batch in enumerate(tqdm(test_dataloader_poison)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits1 = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = logits1.logits
            predictions = logits.argmax(dim=1)
            predictions, references = predictions.to('cpu'), batch["labels"].to('cpu')
            correct = (predictions == references).sum().item()
            total_correct_test += correct
            total_number_test += references.size(0)
        torch.cuda.empty_cache()
        test_poison_acc = total_correct_test / total_number_test
        asr = 1.0 - test_poison_acc
        return test_clean_acc, asr

    # test for our method
    test_clean_acc, asr = dev_for_acc_asr(test_dataloader, test_dataloader_poison)
    print('Our Method Accuracy: %.4f' % test_clean_acc)
    print('Our Method ASR: %.4f' % asr)

if __name__ == "__main__":
    main()
