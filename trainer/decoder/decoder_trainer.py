import torch
import torch.nn as nn
import numpy as np
from trainer.basic_evaluator import EvaluatorBasic
from trainer.basic_trainer import TrainerBasic
from configs.config import Config
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from utils.evaluate_utils import calculate
from utils.json_utils import load_json, write_json
from sklearn.metrics import f1_score, precision_score, recall_score
from torchviz import make_dot

class DecoderTrainer(TrainerBasic):
    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, dev_dataloader: DataLoader = None, evaluator: EvaluatorBasic = None) -> None:
        super().__init__(config, model, train_dataloader, test_dataloader, dev_dataloader, evaluator)

    def one_step(self, data) -> Tuple[torch.Tensor, tuple]:
        token_ids, entities_ids, summar_ids, bertsum_ids, token_mask, entities_mask, summar_mask, bertsum_mask, role_names, role_labels, role_spans, entity_span, char2token, entity2token = data
        '''
        token_ids [batch, seq_len]
        summar_ids [batch, seq_len]
        bertsum_ids [batch, seq_len]
        entities_ids [batch, seq_len]
        roles [batch, arg_roles]
        role_labels [batch, seq_len]
        entity_span[batch, entity_num, multi_span]
        '''
        # 模型训练
        # try:
        loss, module_output = self.model(token_ids, entities_ids, summar_ids, bertsum_ids, token_mask, entities_mask, summar_mask, bertsum_mask, role_labels, entity_span, char2token, entity2token)
        # make_dot(module_output, params=dict(list(self.model.named_parameters()))).render("torchviz", format="png")
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        return loss, module_output

    def init_log_cache(self) -> Dict[str, list]:
        log_cache = {
            "loss": [],
            "true_args": [],
            "pred_args" : [],
            "role_names": []
        }
        return log_cache

    def update_log_cache(self, log_cache: dict, data, loss, model_output, tag):
        '''
            存储模型输出结果
        '''
        super().update_log_cache(log_cache, data, loss, model_output)
        
        pred_args = model_output.transpose(0,1) # [batch_size* role_num * seq_len]
        pred_args = self.vec2span(pred_args, data[-3], data[-2])
        true_args = data[-4] # [batch_size* arg]
        log_cache['true_args'].extend(true_args)
        log_cache['role_names'].extend(data[-6])
        log_cache['pred_args'].extend(pred_args)


    def calculate_matrics_and_save_log(self, log_cache, tag: str):
        '''
            根据存储的模型结果计算指标
        '''
        if tag == 'train':
            dict_lr = {
                'lr': self.scheduler.get_last_lr()[0]
            }
            self.config.tbWriter.add_scalars('lr', dict_lr, global_step=self.total_batch)
            
        dict_loss = {
            tag+'_loss': np.array(log_cache['loss']).mean()
        }
        self.config.tbWriter.add_scalars('loss', dict_loss, global_step=self.total_batch)
        
        sent = {
                'f1': 0,
                'precision': 0,
                'recall': 0
            }
        
        # 计算评估指标
        if tag != 'train':
            metric_score = calculate(self.config, log_cache['pred_args'], log_cache['true_args'], log_cache['role_names'])
            
            sent['f1'] = metric_score['f_c']
            sent['precision'] = metric_score['p_c']
            sent['recall'] = metric_score['r_c']

            self.config.tbWriter.add_scalars(
                'sent', sent, global_step=self.total_batch)
            self.config.logger.info('Test loss: {:.2f}'.format(np.array(log_cache['loss']).mean()))
            self.config.logger.info("---------------------------------------------------------------------")
            self.config.logger.info('Arg      - P: {:6.2f}            , R: {:6.2f}            , F: {:6.2f}'.format(
                    sent['precision'] * 100.0,sent['recall'] * 100.0,  sent['f1'] * 100.0))
            self.config.logger.info("---------------------------------------------------------------------")
            self.write_rams_result(log_cache['role_names'], log_cache['true_args'], log_cache['pred_args'])

        # 清空 cache
        log_cache_init = self.init_log_cache()
        for key in log_cache.keys():
            log_cache[key] = log_cache_init[key]

        return sent['f1']
    
    
    def write_rams_result(self, role_names, true_args, pred_args):
        docs = load_json(self.config.test_path)
        result = []
        for i, doc in enumerate(docs):
            doc_key = doc['doc_key']
            sentences = doc['sentences']
            text = []
            role_with_span = {}
            for sent in sentences:
                text.extend(sent)
            roles = role_names[i]
            true_arg = true_args[i]    
            pred_arg = pred_args[i]
            for j, role in enumerate(roles):
                true_arg_span = true_arg[j]
                pred_arg_span = pred_arg[j]
                true_arg_word = set()
                pred_arg_word = set()
                for span in true_arg_span:
                    true_arg_word.update(text[span[0]:span[1]])
                for span in pred_arg_span:
                    pred_arg_word.update(text[span[0]:span[1]])   
                role_with_span[role] = {'true': list(true_arg_word), 'pred': list(pred_arg_word)}
                # true_arg_idx = [idx for idx, k in enumerate(true_arg[j]) if k>0]
                # pred_arg_idx = [idx for idx, k in enumerate(pred_arg[j]) if k>0]
                # true[role] = [[idx, text[idx]] for idx in true_arg_idx]
                # pred[role] = [[idx, text[idx]] for idx in pred_arg_idx]
            result.append({'doc_key': doc_key,'sentence': ' '.join(text), 'role_with_span': role_with_span})
        write_json(result, 'result.json')
        
    def vec2span(self, pred_args, entity_spans, char2token):
        # [batch_size* role_num * seq_len]
        pred_arg_spans = []
        batch_size = len(pred_args)
        for i in range(batch_size):
            roles = pred_args[i] # 一个文档中的所有角色得分
            entity_span = entity_spans[i] #一个文档中的所有实体span
            pred_arg_span = [] #一个文档所有论元角色的所有span
            for vec in roles:
                role_span = set()# 一个论元角色的所有span
                max_pos = 0
                max_logits = 0 
                for j, logits in enumerate(vec):
                    if logits > max_logits:
                        max_pos = j
                        max_logits = logits
                    # if logic > self.config.threshold:
                    #     span = self.judge_id_in_span(j, entity_span)
                    #     if span:
                    #         role_span.update(span)         
                    #     else:
                    #         role_span.add((j,j+1))
                if max_logits > self.config.threshold:
                    span = self.judge_id_in_span(max_pos, entity_span)
                    if span:
                        start = char2token[i][span[0]]
                        end = char2token[i][span[1]]
                        end_ = char2token[i][span[1]-1] #以防end 映射到None但end-1映射有值
                        if start and end:
                            end = end if end > start else start+1
                            role_span.add((start, end))
                        elif start and end_:
                            role_span.add((start, end_ + 1))
                    else:
                        start = char2token[i][max_pos]
                        if start:
                            end = start + 1
                            role_span.add((start, end))
                pred_arg_span.append(list(role_span))
            pred_arg_spans.append(pred_arg_span)
        return pred_arg_spans

    # 如果一个位置的概率超过阈值，将包含该次的实体以及该实体的所有span添加
    # def judge_id_in_span(self, id, entity_span):
    #     for entity in entity_span:
    #         for span in entity:
    #             if id in range(span[0], span[1]):
    #                 return entity
    #     return None
    
    def judge_id_in_span(self, id, entity_span):
        for entity in entity_span:
            for span in entity:
                if id in range(span[0], span[1]):
                    return span
        return None
    