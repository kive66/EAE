import torch
import torch.nn as nn
from transformers import AutoModel
from configs.config import Config

class RoleDecoder(nn.Module):
    '''
        论元角色融合解码器    
    '''
    def __init__(self, config: Config):
        super(RoleDecoder, self).__init__()

        self.config = config
        self.single_linear = nn.Linear(self.config.hidden_size*3, 1)
        self.multi_linear = nn.Linear(self.config.hidden_size*3, 1)
        self.answer_linear = nn.Linear(self.config.hidden_size*2, self.config.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.cal_loss = nn.BCELoss() 
             
    def forward(self, role_labels, summar_role_embedding, token_embedding, entities_embedding, token_mask, entity_mask, entity_spans, char2token, entity2token):
        '''
            role_ids: 论元角色编码 [role_num * batch * seq_len]
            role_labels: 论元角色真实标签 [role_num * batch * seq_len]
            summar_embedding: 摘要编码 [batch * seq_len * hidden_size]
            token_embedding: 句子编码 [batch * seq_len * hidden_size]
            entities_embedding: 句子所有实体相连接的编码 [batch * seq_len * hidden_size]
            entity_spans: 实体在文档中的位置
            char2token: 原始分词与bert分词的映射
        '''
        batch_arg_logits = []
        total_loss = torch.tensor(0.0).to(self.config.device)
        pre_answer = torch.zeros_like(token_embedding).to(self.config.device)

        # 遍历batch中每个doc的第i个论元
        for i, sr_embedding in enumerate(summar_role_embedding):
            gold_labels = role_labels[i]
            # summar + role + candi_answer + pre_answer
            # print(summar_embedding.shape)
            # print(role_embedding.shape)
            # print(token_embedding.shape)
            # print(pre_answer.shape)
            
            single_word_embedding = self.single_linear(torch.cat((sr_embedding, token_embedding, pre_answer), dim=-1))
            multi_word_embedding = self.multi_linear(torch.cat((sr_embedding, entities_embedding, pre_answer), dim=-1))
            
            single_word_pred_arg = self.sigmoid(single_word_embedding).squeeze(-1) # 根据token的单字分类结果 (batch * seqlen)
            multi_word_pred_arg = self.sigmoid(multi_word_embedding).squeeze(-1) # 根据entity的多姿分类结果 (batch * seqlen)
            
            pred_arg_logits = self.arg_map(single_word_pred_arg, multi_word_pred_arg, entity_spans, char2token, entity2token)
            batch_arg_logits.append(pred_arg_logits)
            
            curr_answer = torch.mul(token_embedding, pred_arg_logits.unsqueeze(-1))
            pre_answer = self.answer_linear(torch.cat((curr_answer, pre_answer), dim=-1))
            
            loss = self.cal_loss(pred_arg_logits, gold_labels)
            loss = (loss*token_mask.float()).sum()
            total_loss += loss
            
        return total_loss.mean(), torch.stack(batch_arg_logits)
    
    
    def arg_map(self,single_word_pred, multi_word_pred, entity_spans, char2token, entity2token):
        '''
            映射token到原始文档位置
            映射entity到原始文档位置
            entity_spans: batch[ entities[ multi_span[ [start, end ], ... ], ... ], ... ]
        '''
        # pred_arg_ids = [] # 按照原句分词映射的结果
        pred_arg_logits= [] # 按照bert分词映射的结果
        batch_size = len(char2token)
        for i in range(batch_size):
            pred_token_logits = single_word_pred[i] # 按照原句分词的token映射结果
            pred_entity_logits = torch.zeros(self.config.max_seq_len).to(self.config.device) # 按照原句分词的entity映射结果
            # pred_arg_logits = torch.zeros(self.config.max_seq_len).to(self.config.device) #按照bert分词的融合结果
            
            # 映射token embedding 到原句
            # word_ids = char2token[i]
            # for j in range(len(word_ids)):
            #     if word_ids[j] != None:
            #         word_pos = word_ids[j]
            #         pred_token_ids[word_pos] = max(pred_token_ids[word_pos], single_word_pred[i][j])
                
            # 映射entity embedding到entity
            word_ids = entity2token[i]
            entity_scores = torch.zeros(len(entity_spans[i]))
            for j in range(len(word_ids)):
                if word_ids[j] != None:
                    word_pos = word_ids[j]
                    if word_pos >= len(entity_spans[i]):
                        break 
                    entity_scores[word_pos] = max(entity_scores[word_pos], multi_word_pred[i][j])
            
            #映射entity到token embedding该entity所有span上
            for j, entity in enumerate(entity_spans[i]):
                for span in entity:
                    for k in range(span[0], span[1]):
                        pred_entity_logits[k] = entity_scores[j]
                
            merged_pred_arg = torch.cat((pred_token_logits.unsqueeze(-1), pred_entity_logits.unsqueeze(-1)),dim =-1)
            merged_pred_arg = torch.max(merged_pred_arg, dim=-1)[0]
            
            # word_ids = char2token[i]
            # for j in range(len(word_ids)):
            #     if word_ids[j] != None:
            #         pred_arg_embedding[j] = merged_pred_arg[word_ids[j]]
                
            pred_arg_logits.append(merged_pred_arg)
        return torch.stack(pred_arg_logits).to(self.config.device)
    