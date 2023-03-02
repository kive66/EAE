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
        self.single_linear = nn.Sequential(
            nn.Dropout(self.config.drop_rate),
            nn.Linear(self.config.hidden_size*3+1, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.drop_rate),
            nn.Linear(self.config.hidden_size, 1)
        )
        self.multi_linear = nn.Sequential(
            nn.Dropout(self.config.drop_rate),
            nn.Linear(self.config.hidden_size*3+1, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.drop_rate),
            nn.Linear(self.config.hidden_size, 1)
        )
        self.answer_linear = nn.Linear(self.config.max_seq_len*2, self.config.max_seq_len)
        self.token_sigmoid = nn.Sigmoid()
        self.entity_sigmoid = nn.Sigmoid()
        self.cal_loss = nn.BCELoss()   
             
    def forward(self, bert, role_ids, role_labels, summar_embedding, token_embedding, entities_embedding, entity_spans, token_mask, role_ids_mask, char2token, span2entity):
        '''
            role_ids: 论元角色编码 [role_num * batch * seq_len]
            role_labels: 论元角色真实标签 [role_num * batch * seq_len]
            summar_embedding: 摘要编码 [batch * seq_len * hidden_size]
            token_embedding: 句子编码 [batch * seq_len * hidden_size]
            entities_embedding: 句子所有实体相连接的编码 [batch * seq_len * hidden_size]
            entity_spans: 实体在文档中的位置
            char2token: 原始分词与bert分词的映射
        '''
        pred_arg_ids = []
        total_loss = torch.tensor(0.0).to(self.config.device)
        pre_answer = torch.zeros_like(token_mask).to(self.config.device).unsqueeze(-1)

        # 遍历batch中每个doc的第i个论元
        for i, role_id in enumerate(role_ids):
            gold_labels = role_labels[i]
            role_mask = role_ids_mask[i]
            role_embedding = bert(role_id, role_mask)[0]
            
            # summar + role + candi_answer + pre_answer
            single_word_embedding = self.single_linear(torch.cat((summar_embedding, role_embedding, token_embedding, pre_answer), dim=-1))
            multi_word_embedding = self.multi_linear(torch.cat((summar_embedding, role_embedding, entities_embedding, pre_answer), dim=-1))
            
            single_word_pred_arg = self.token_sigmoid(single_word_embedding).squeeze(-1) # 根据token的单字分类结果 (batch * seqlen)
            multi_word_pred_arg = self.entity_sigmoid(multi_word_embedding).squeeze(-1) # 根据entity的多姿分类结果 (batch * seqlen)
            
            pred_arg_ids_ = self.arg_map(single_word_pred_arg, multi_word_pred_arg, entity_spans, char2token, span2entity)
            pre_answer = self.answer_linear(torch.cat((pred_arg_ids_, pre_answer.squeeze(-1)), dim=-1)).unsqueeze(-1)
            pred_arg_ids.append(pred_arg_ids_)
            
            loss = self.cal_loss(pred_arg_ids_, gold_labels)
            loss = (loss*token_mask.float()).sum()
            total_loss += loss
            
        return total_loss, torch.stack(pred_arg_ids)
    
    
    def arg_map(self,single_word_pred, multi_word_pred, entity_spans, char2token, span2entity):
        '''
            映射token到原始文档位置
            映射entity到原始文档位置
            entity_spans: batch[ entities[ multi_span[ [start, end ], ... ], ... ], ... ]
        '''
        pred_arg_ids = []
        batch_size = len(char2token)
        for i in range(batch_size):
            pred_token_ids = torch.zeros(self.config.max_seq_len).to(self.config.device)
            pred_entity_ids = torch.zeros(self.config.max_seq_len).to(self.config.device)
            # 映射token
            for j, c2t in enumerate(char2token[i]):
                pred_token_ids[j] = torch.max(single_word_pred[i][c2t[0]:c2t[1]])# 映射token到原句
                
                # # 提取token对应span
                # if pred_token_ids[j] > self.config.threshold:
                #     in_span = []
                #     for entity in entity_spans[i]:
                #         for span in entity:
                #             if j>= span[0] and j< span[1]:
                #                 pred_arg_span.append([j,j+1])
                #     pred_arg_span.add([j,j+1])
            
            # 映射span
            for j in range(len(span2entity)):
                s2e = span2entity[i][j] # batch i entity j    
                entity = entity_spans[i][j] # batch i entity j   
                char_scores = [] # entity编码中 一个 entity 每个 char 的score
                # 提取entity编码中 每个char的概率
                for span in s2e:
                    char_score = torch.max(multi_word_pred[i][span[0]:span[1]])
                    char_scores.append(char_score)
                # 将entity每个char映射到原始句子位置中（一个entity可能在原句中有多个span）
                char_scores = torch.stack(char_scores).to(self.config.device)
                for span in entity:
                    if span[1] <= self.config.max_seq_len:
                        pred_entity_ids[span[0]:span[1]] = char_scores
                
                # if max(char_score) > self.config.threshold:
                #     for span in entity:
                #         pred_arg_span.add(span)
            
            pred_arg = torch.div(torch.add(pred_token_ids, pred_entity_ids), 2)
            pred_arg_ids.append(pred_arg)
        return torch.stack(pred_arg_ids)
    