import torch
import torch.nn as nn
from transformers import AutoModel
from configs.config import Config
from models.layers.summar_encoder import SummarEncoder
from models.layers.sigmoid_role_decoder import RoleDecoder


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()
        self.config = config
        
        self.bert = AutoModel.from_pretrained(config.pretrain_path, output_hidden_states=True)
        self.bert.resize_token_embeddings(len(config.tokenizer))
        
        self.sum_encoder = SummarEncoder(config) # 对doc摘要进行编码
        self.role_decoder = RoleDecoder(config) # 对role进行预测
        
                
    def forward(self, token_ids, entities_ids, summar_ids, bertsum_ids, token_mask, entities_mask, summar_mask, bertsum_mask, role_labels, entity_spans, char2token, entity2token):
        token_embedding = self.bert(token_ids, token_mask)[0]
        entities_embedding = self.bert(entities_ids, entities_mask)[0]
        summar_embedding = self.sum_encoder(self.bert, summar_ids, bertsum_ids, summar_mask, bertsum_mask)
        
        # 根据论元角色顺序进行双向预测
        forward_loss, forward_ids = self.role_decoder(role_labels, summar_embedding, token_embedding, entities_embedding, entity_spans, token_mask, char2token, entity2token)

        reversed_summar_role_embedding = torch.flip(summar_embedding, [0])
        reversed_role_labels = torch.flip(role_labels, [0])
        
        backward_loss, backward_ids = self.role_decoder(reversed_role_labels, reversed_summar_role_embedding, token_embedding, entities_embedding, entity_spans, token_mask, char2token, entity2token)
        
        best_ids = self.get_best_pred(forward_ids, backward_ids)
        total_loss = forward_loss + backward_loss
        
        return total_loss.mean(), best_ids
    

    def get_best_pred(self, forward_ids, backward_ids):
        backward_ids = torch.flip(backward_ids, [0])
        
        merge_ids = torch.cat((forward_ids.unsqueeze(-1), backward_ids.unsqueeze(-1)),dim =-1)
        best_ids = torch.max(merge_ids, dim=-1)[0]
        # result_spans = forward_spans
        # role_num = self.config.max_role_num
        # for i in range(role_num):
        #     for j in range(batch_size):
        #         result_spans[i][j].extend(backward_spans[role_num-i-1[j]])
        return best_ids
            