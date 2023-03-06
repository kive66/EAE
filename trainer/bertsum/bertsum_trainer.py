from trainer.basic_evaluator import EvaluatorBasic
from trainer.basic_trainer import TrainerBasic
from configs.config import Config
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class BertSumTrainer(TrainerBasic):
    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, dev_dataloader: DataLoader = None, evaluator: EvaluatorBasic = None) -> None:
        super().__init__(config, model, train_dataloader, test_dataloader, dev_dataloader, evaluator)

    def one_step(self, data) -> Tuple[torch.Tensor, tuple]:
        src, labels, segs, clss, mask, mask_cls, src_str, tgt_str, summarization = data
        '''
        src [batch, seq_len]
        labels [batch, sent_num]
        segs [batch, seq_len]
        clss [batch, sent_num]
        mask [batch, seq_len]
        mask_cls [batch, seq_len]
        '''
        # 模型训练
        loss, sent_pred = self.model(src, labels, segs, clss, mask, mask_cls, summarization)
        return loss, (sent_pred)

    def init_log_cache(self) -> Dict[str, list]:
        log_cache = {
            "loss": [],
            "sent_true": [],
            "sent_pred" : [],
        }
        return log_cache

    def update_log_cache(self, log_cache: dict, data, loss, model_output):
        # 先存loss
        super().update_log_cache(log_cache, data, loss, model_output)
        sent_pred = model_output
        sent_true = data[1]
        sent_pred[sent_pred > self.config.threshold] = 1
        sent_pred[sent_pred <= self.config.threshold] = 0
        log_cache['sent_pred'].extend(sent_pred.int().detach().cpu().tolist())
        log_cache['sent_true'].extend(sent_true.int().detach().cpu().tolist())


    def calculate_matrics_and_save_log(self, log_cache, tag: str):
        if tag == 'train':
            dict_lr = {
                'lr': self.scheduler.get_last_lr()[0]
            }
            self.config.tbWriter.add_scalars('lr', dict_lr, global_step=self.total_batch)
            
        dict_loss = {
            tag+'_loss': np.array(log_cache['loss']).mean()
        }
        self.config.tbWriter.add_scalars('loss', dict_loss, global_step=self.total_batch)

        # 计算评估指标
        p, r, f  = self.metric(log_cache['sent_true'], log_cache['sent_pred'])
        sent = {
            'f1': f,
            'precision': p,
            'recall': r
        }
        self.config.tbWriter.add_scalars(
            'sent', sent, global_step=self.total_batch)

        if tag!= 'train':
            self.config.logger.info('Test loss: {:.2f}'.format(np.array(log_cache['loss']).mean()))
            self.config.logger.info("---------------------------------------------------------------------")
            self.config.logger.info('Sent      - P: {:6.2f}            , R: {:6.2f}            , F: {:6.2f}'.format(
                    sent['precision'] * 100.0,sent['recall'] * 100.0,  sent['f1'] * 100.0))
            self.config.logger.info("---------------------------------------------------------------------")

        # 清空 cache
        log_cache_init = self.init_log_cache()
        for key in log_cache.keys():
            log_cache[key] = log_cache_init[key]
        if r> 0.6 and p>0.5:
            return 0.6*r +0.4*p
        else:
            return 0
            

    def metric(self, y_gold, y_pred):
        true_positive = 0
        pred_positive, gold_positive = 0,0
        for i in range(len(y_gold)):
            for j in range(len(y_gold[i])):
                gold_positive += y_gold[i][j]
                pred_positive += y_pred[i][j]
                if y_gold[i][j] and y_pred[i][j]:
                    true_positive += 1

        prec_c, recall_c, f1_c = 0, 0, 0
        if pred_positive != 0:
            prec_c = true_positive / pred_positive
        else:
            prec_c = 0
        if gold_positive != 0:
            recall_c = true_positive / gold_positive
        else:
            recall_c = 0
        if prec_c or recall_c:
            f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
        else:
            f1_c = 0
        return prec_c, recall_c, f1_c
