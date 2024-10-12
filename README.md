## Introduction
Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation

## Requirements
* Python == 3.8.19
* torch == 2.2.2+cu118
* transformers == 4.40.2
* accelerate == 0.30.0
* deepspeed == 0.15.1 

## Weak-to-Strong Unlearning Backdoor 

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml context_learning.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_clean_sentence.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_sentence.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_clean_prompt.py --model facebook/opt-1.3b
```

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29528 --config_file run.yaml attack_prompt.py --model facebook/opt-1.3b
```

## Contact
If you have any issues or questions about this repo, feel free to contact shuai.zhao@ntu.edu.sg.
