from trainer.basic_evaluator import EvaluatorBasic
from trainer.basic_trainer import TrainerBasic
from configs.config import Config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import copy

class BertSumEvalutor(EvaluatorBasic, TrainerBasic):
    def __init__(self, config: Config, model: nn.Module) -> None:
        super().__init__(config, model)

    def one_step(self, data) -> Tuple[torch.Tensor, tuple]:
        src, labels, segs, clss, mask, mask_cls, src_txt, tgt_txt, summarization = data
        '''
        input_ids [batch, seq_len]
        attention_mask [batch, seq_len]
        event_type [batch]
        trigger_start [batch]
        trigger_end [batch]
        '''
        # 模型训练
        loss, sent_pred = self.model(src, labels, segs, clss, mask, mask_cls, summarization)
        '''
        也可以不拆包，直接写
        model_output = self.model(*data)
        '''
        return loss.mean(), sent_pred

    def convert_token_to_char(self,offset, start_token_index, end_token_index):
        return offset[start_token_index][0], offset[end_token_index][1]

    @torch.no_grad()
    def evaluate(self, datloader: DataLoader, tag: str, total_step=None) -> float:
        data_all = []
        model_ouput_all = []
        loss_all = []
        for i, data in enumerate(tqdm(datloader)):
            # 存储之间原本结果
            data_all.append(data)
            data = self.to_device(data, self.config)
            loss, sent_pred = self.one_step(data)
            loss_all.append(loss.mean().item())
            # 模型输出解包
            sent_pred[sent_pred > self.config.threshold] = 1
            sent_pred[sent_pred <= self.config.threshold] = 0

            model_ouput_all.append(sent_pred)        
        loss = np.array(loss_all).mean()
        inference_results = self.inference(data_all,model_ouput_all)
        return loss, inference_results

    def inference(self,data_all, model_ouput_all):
        inference_results = []
        evaluate_results = []
        for (data, output) in tqdm(zip(data_all, model_ouput_all)):
            src, labels, segs, clss, mask, mask_cls, src_str, tgt_str, summarization = data
            sent_pred = output
            summarization = ''
            for i, sent in enumerate(src_str[0]):
                if sent_pred[0][i]:
                    summarization += sent

            inference_results.append(summarization)
        return inference_results
