import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2, new_hidden_size=4096):
        super(CustomBertForSequenceClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=True)
        original_hidden_size = self.bert.config.hidden_size
        self.hidden_layer = nn.Linear(original_hidden_size, new_hidden_size)
        self.activation = nn.ReLU()  # 使用 ReLU 作为激活函数

        # 最终的分类器从 new_hidden_size 映射到 num_labels
        self.classifier = nn.Linear(new_hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 从 BertForSequenceClassification 获取输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_hidden_states=True)
        sequence_output = outputs.hidden_states[-1][:, 0]

        # 通过新增的隐藏层
        hidden_output = self.hidden_layer(sequence_output)
        transformed_hidden_output = self.activation(hidden_output)  # 应用激活函数

        # 通过分类器得到 logits
        logits = self.classifier(transformed_hidden_output)

        output = {'logits': logits, 'hidden_states': transformed_hidden_output}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output['loss'] = loss

        return output