
import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn as nn
from configs.config import Config


class Summarizer(nn.Module):
    def __init__(self, config: Config):
        super(Summarizer, self).__init__()

        self.config = config
        self.bert = AutoModel.from_pretrained(config.pretrain_path)
        self.bert.resize_token_embeddings(len(config.tokenizer))

        self.fc = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()

    def forward(self, sent_ids, labels, segs, clss, mask, mask_cls, summarization):
        # 拿出文本表征
        sent_embedding = self.bert(sent_ids, mask, token_type_ids = segs)
        # sum_embedding = self.bert(summarization.input_ids, summarization.attention_mask)
        sent_last_hidden = sent_embedding[0]
        sents_vec = sent_last_hidden[torch.arange(sent_last_hidden.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        loss = self.loss(sent_scores, labels.float())
        loss = (loss*mask.float()).sum()
        return loss, sent_scores 
        

# class Bert(nn.Module):
#     def __init__(self, temp_dir, load_pretrained_bert, bert_config):
#         super(Bert, self).__init__()
#         if(load_pretrained_bert):
#             # self.model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
#             self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
#         else:
#             self.model = BertModel(bert_config)

#     def forward(self, x, segs, mask):
#         encoded_layers, _ = self.model(x, segs, attention_mask =mask)
#         top_vec = encoded_layers[-1]
#         return top_vec



# class Summarizer(nn.Module):
#     def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
#         super(Summarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        

#     def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):

#         top_vec = self.bert(x, segs, mask)
#         sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
#         sents_vec = sents_vec * mask_cls[:, :, None].float()
#         sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
#         return sent_scores, mask_cls

# class Classifier(nn.Module):
#     def __init__(self, hidden_size):
#         super(Classifier, self).__init__()
#         self.linear1 = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, mask_cls):
#         h = self.linear1(x).squeeze(-1)
#         sent_scores = self.sigmoid(h) * mask_cls.float()
#         return sent_scores
