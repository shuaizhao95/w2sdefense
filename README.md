## Introduction
Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation

## Requirements
* Python == 3.8.19
* torch == 2.2.2+cu118
* transformers == 4.40.2
* accelerate == 0.30.0
* deepspeed == 0.15.1 

## Weak-to-Strong Unlearning Backdoor 

Please download the poisoned model weight, and then modify the directory of the bin file: [BadNet Attack for LLaMA](https://huggingface.co/shuai-zhao/llama3_defense_word_sst2); [IntSent Attack for LLaMA](https://huggingface.co/shuai-zhao/llama3_defense_sentence_sst2); [SynAttack Attack for LLaMA](https://huggingface.co/shuai-zhao/llama_defense_synattack_sst2).

[Please download the clean teacher model] (https://huggingface.co/shuai-zhao/llama3_defense_word_sst2)

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
