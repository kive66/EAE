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
        token_ids, entities_ids, summar_ids, bertsum_ids, token_mask, entities_mask, summar_mask, bertsum_mask, role_names, role_starts, role_ends, role_spans, entity_span, char2token, entity2token = data
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
        loss, module_output = self.model(token_ids, entities_ids, summar_ids, bertsum_ids, token_mask, entities_mask, summar_mask, bertsum_mask, role_starts, role_ends, entity_span, char2token, entity2token)
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
        
        if tag == 'test':
            start_logits, end_logits = model_output
            start_logits = start_logits.transpose(0,1) # [batch_size* role_num * seq_len]
            end_logits = end_logits.transpose(0,1) # [batch_size* role_num * seq_len]
            
            token_ids, entities_ids, summar_ids, bertsum_ids, token_mask, entities_mask, summar_mask, bertsum_mask, role_names, role_starts, role_ends, role_spans, entity_spans, char2token, entity2token = data
            
            pred_args = self.vec2span(start_logits, end_logits, entity_spans, char2token)
            log_cache['true_args'].extend(role_spans)
            log_cache['role_names'].extend(role_names)
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
                    true_arg_word.add(' '.join(text[span[0]:span[1]]))
                for span in pred_arg_span:
                    pred_arg_word.add(' '.join(text[span[0]:span[1]]))
                role_with_span[role] = {'true': list(true_arg_word), 'pred': list(pred_arg_word)}
                # true_arg_idx = [idx for idx, k in enumerate(true_arg[j]) if k>0]
                # pred_arg_idx = [idx for idx, k in enumerate(pred_arg[j]) if k>0]
                # true[role] = [[idx, text[idx]] for idx in true_arg_idx]
                # pred[role] = [[idx, text[idx]] for idx in pred_arg_idx]
            result.append({'doc_key': doc_key,'sentence': ' '.join(text), 'role_with_span': role_with_span})
        write_json(result, 'result.json')
        
    def vec2span(self, batch_start_logits, batch_end_logits, batch_entity_spans, batch_char2token):
        # [batch_size* role_num * seq_len]
        batch_pred_spans = []
        batch_size = len(batch_char2token)
        for i in range(batch_size):
            start_logits = batch_start_logits[i] # 一个文档中的所有角色得分
            end_logits = batch_end_logits[i]
            entity_span = batch_entity_spans[i] #一个文档中的所有实体span
            char2token = batch_char2token[i]
            pred_arg_spans = [] #一个文档所有论元角色的所有span
            for j in range(self.config.max_role_num):
                role_arg_spans = set()
                role_entities = set()
                a = start_logits[j]
                b = end_logits[j]
                role_start_logits, role_start_entities = self.get_best_indexes(start_logits[j], entity_span, larger_than_cls=True, cls_logit=start_logits[j][0])
                role_end_logits, role_end_entities = self.get_best_indexes(end_logits[j], entity_span, larger_than_cls=True, cls_logit=end_logits[j][0])

                role_entities.update(role_start_entities)
                role_entities.update(role_end_entities)
                
                # 将entity span映射回原句
                for span in role_entities:
                    origin_start_index = char2token[span[0]]
                    origin_end_index = char2token[span[1]]
                    origin_end_index_ = char2token[span[1]-1]
                    if origin_start_index and origin_end_index:
                        origin_end_index = origin_end_index if origin_end_index > origin_start_index else origin_start_index + 1
                        span = [origin_start_index, origin_end_index]
                    elif origin_start_index and origin_end_index_:
                        origin_end_index = origin_end_index + 1 if origin_end_index_ > origin_start_index else origin_start_index + 1
                        span = [origin_start_index, origin_end_index]
                    role_arg_spans.add((span[0], span[1]))
                # add span preds
                for start_index in role_start_logits:
                    for end_index in role_end_logits:
                        if end_index <= start_index:
                            continue
                        origin_start_index = char2token[start_index]
                        origin_end_index = char2token[end_index]
                        origin_end_index_ = char2token[end_index-1]
                        if origin_start_index and origin_end_index:
                            length = end_index - start_index + 1
                            if length > 15 :
                                continue
                            origin_end_index = origin_end_index if origin_end_index > origin_start_index else origin_start_index + 1
                            span = [origin_start_index, origin_end_index]
                            role_arg_spans.add((span[0], span[1]))
                        elif origin_start_index and origin_end_index_:
                            origin_end_index = origin_end_index_ + 1 if origin_end_index_ > origin_start_index else origin_start_index + 1
                            span = [origin_start_index, origin_end_index]
                            role_arg_spans.add((span[0], span[1]))
                pred_arg_spans.append(role_arg_spans)
            batch_pred_spans.append(pred_arg_spans)
        return batch_pred_spans

    def get_best_indexes(self, logits, entity_span, n_best_size=1, larger_than_cls=False, cls_logit=None):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        best_entities = set()
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            if larger_than_cls:
                if index_and_score[i][1] < cls_logit:
                    break
            if index_and_score[i][1] > cls_logit*(1+0.8):
                best_indexes.append(index_and_score[i][0])
            else:
                span = self.judge_id_in_span(index_and_score[i][0], entity_span)
                best_entities.add(span)
        return best_indexes, best_entities
    
    def judge_id_in_span(self, id, entity_span):
        for entity in entity_span:
            for span in entity:
                if id in range(span[0], span[1]):
                    return span
        return None
    