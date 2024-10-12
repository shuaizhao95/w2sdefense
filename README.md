## Introduction
Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation

## Requirements
* Python == 3.8.19
* torch == 2.2.2+cu118
* transformers == 4.40.2
* accelerate == 0.30.0
* deepspeed == 0.15.1 

## Weak-to-Strong Unlearning Backdoor 

[链接文本](URL "可选的标题")
```shell
cd word # download poisoned model weight.
```

```shell
DS_SKIP_CUDA_CHECK=1 python lora.py --model_name_or_path meta-llama/Meta-Llama-3-8B --poison word
```

```shell
DS_SKIP_CUDA_CHECK=1 python unlearning.py --model_name_or_path meta-llama/Meta-Llama-3-8B --poison word
```

```shell
DS_SKIP_CUDA_CHECK=1 python test.py --model_name_or_path meta-llama/Meta-Llama-3-8B --poison word
```


## Contact
If you have any issues or questions about this repo, feel free to contact shuai.zhao@ntu.edu.sg.
