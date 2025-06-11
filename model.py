import torch
import torch.nn as nn
from transformers import BertModel

class HybridModel(nn.Module):
    def __init__(self, audio_embedding_dim, num_labels=5, bert_model_name='bert-large-uncased'):
        super(HybridModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.audio_processor = nn.Linear(audio_embedding_dim, 256)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 256, num_labels)
        self.text_only_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, audio_embeddings, has_audio):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = []

        for idx, feat in enumerate(pooled):
            if has_audio[idx] == 1:
                audio_feat = self.audio_processor(audio_embeddings[idx].unsqueeze(0))
                combined = torch.cat((feat.unsqueeze(0), audio_feat), dim=1)
                logits.append(self.classifier(combined))
            else:
                logits.append(self.text_only_classifier(feat.unsqueeze(0)))

        return torch.cat(logits, dim=0)