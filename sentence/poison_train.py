import argparse

import evaluate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import (PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel, PeftConfig)
from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftType, PrefixTuningConfig, PromptEncoderConfig)
from peft.utils.other import fsdp_auto_wrap_policy
from data import data_load_poison
import os
import logging
import sys
from log import LoggerWriter
# 重定向标准输出和错误输出
def parse_args():
    parser = argparse.ArgumentParser(description="PEFT a transformers model on a sequence classification task")
    parser.add_argument("--model_name_or_path", type=str, default='robert')
    parser.add_argument("--poison", type=str, default=None, help="poison method.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--data_path", type=str, default='./data/sst-2', help="Data path.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--num_warmup_steps", type=int, default=0,help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    tokenizer_kwargs = {}

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "llama", "vicuna", "mistral", "qwen")):
        tokenizer_kwargs["padding_side"] = "left"
    else:
        tokenizer_kwargs["padding_side"] = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    metric = evaluate.load("glue", 'sst2')

    train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison = data_load_poison(tokenizer, args.poison, accelerator, args.per_device_train_batch_size, args.per_device_eval_batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16)#('model_save')

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
        model = accelerator.prepare(model)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=args.num_warmup_steps,num_training_steps=(len(train_dataloader) * args.num_train_epochs))

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison, optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison, optimizer, lr_scheduler)
    else:
        model, train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, eval_dataloader, test_dataloader, test_dataloader_poison, optimizer, lr_scheduler)

    def evaluation_dev(model, eval_dataloader):
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        torch.cuda.empty_cache()
        return eval_metric

    best_dev_acc = -1
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        eval_metric = evaluation_dev(model, eval_dataloader)
        dev_clean_acc = eval_metric['accuracy']
        accelerator.print(f"epoch {epoch}:", eval_metric['accuracy'])
        if dev_clean_acc > best_dev_acc:
            best_dev_acc = dev_clean_acc

            test_metric = evaluation_dev(model, test_dataloader)
            accelerator.print(f"CA test:", test_metric['accuracy'])

            test_metric = evaluation_dev(model, test_dataloader_poison)
            accelerator.print(f"ASR test:", 1.0 - test_metric['accuracy'])

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), './model_poisoned/pytorch_model.bin')
            if (1.0 - test_metric['accuracy']) > 0.95:
                exit()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
