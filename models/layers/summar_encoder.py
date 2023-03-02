import torch
import torch.nn as nn
from transformers import AutoModel
from configs.config import Config

class SummarEncoder(nn.Module):
    def __init__(self, config: Config):
        super(SummarEncoder, self).__init__()

        self.config = config
        self.gate = Gate(config.hidden_size)
        
    def forward(self, bert, summar_ids, bertsum_ids, summar_mask, bertsum_mask):
        '''
            summar: t5摘要
            bertsum: bertsum摘要
        '''
        summar_embedding = bert(summar_ids, summar_mask)
        bertsum_embedding = bert(bertsum_ids, bertsum_mask)
        avg_summar = self.get_last_k_hidden(summar_embedding[2]) # [batch * hidden_size]
        avg_bertsum = self.get_last_k_hidden(bertsum_embedding[2]) # [batch * hidden_size]
        summar_embedding = self.gate(avg_summar, avg_bertsum)
        return summar_embedding
    
    def get_last_k_hidden(self, hidden_layer, k=3):
        tmp = 0
        for i in range(k):
            tmp += hidden_layer[-i]
        avg_hidden_layer = tmp/k
        return avg_hidden_layer
        
        sum + role_prompt + vocab
          
class Gate(nn.Module):
    def __init__(self, embed_dim: int):
        super(Gate, self).__init__()
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim)
        self._init_linear()

    def _init_linear(self) -> None:
        nn.init.xavier_normal_(self.gate_linear.weight)
        self.gate_linear.bias.data.fill_(0)

    def forward(self, pooled, feature):
        h = torch.cat((pooled, feature), dim=-1)
        gate = torch.sigmoid(self.gate_linear(h))
        return gate * pooled + (1 - gate) * feature