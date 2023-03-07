from typing import Dict, Tuple
from configs.config import Config
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import timedelta

from trainer.basic_evaluator import EvaluatorBasic

class TrainerBasic():
    def __init__(self, config:Config, model:nn.Module, train_dataloader:DataLoader, test_dataloader:DataLoader, dev_dataloader:DataLoader=None, evaluator:EvaluatorBasic=None) -> None:
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dev_dataloader = dev_dataloader

        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_schedule()

        self.total_batch = 0
        self.best_metices = -1e9
        self.last_improve = 0

        self.evaluator = evaluator

    ###############  自定义优化策略  ###############
    def init_optimizer(self):
        '''
            初始化优化器
        '''
        no_decay = ['bias', 'LayerNorm.weight']
        pretrained_model_prefix = 'encoder'
        list_param = [
            {'params': [param for name, param in self.model.named_parameters() if pretrained_model_prefix in name and any(nd in name for nd in no_decay)], 'lr':self.config.encoder_learning_rate, 'weight_decay':0.0},
            {'params': [param for name, param in self.model.named_parameters() if pretrained_model_prefix in name and not any(nd in name for nd in no_decay)], 'lr':self.config.encoder_learning_rate, 'weight_decay':0.01},
            {'params': [param for name, param in self.model.named_parameters() if pretrained_model_prefix not in name and any(nd in name for nd in no_decay)], 'lr':self.config.basic_learning_rate, 'weight_decay':0.0},
            {'params': [param for name, param in self.model.named_parameters() if pretrained_model_prefix not in name and not any(nd in name for nd in no_decay)], 'lr':self.config.basic_learning_rate, 'weight_decay':0.01},
        ]
        optimizer = AdamW(list_param)

        return optimizer

    def init_schedule(self):
        '''
            初始化lr schedule
        '''
        return get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = int(self.config.num_epochs * len(self.train_dataloader) * self.config.rate_warmup_steps), 
            num_training_steps = self.config.num_epochs * len(self.train_dataloader),
        )
    ###############  自定义优化策略  ###############



    ###############  定义数据如何输入模型，如何得到loss  ###############
    def one_step(self, data) -> Tuple[torch.Tensor, tuple]:
        '''
            data: collate_fn 的输出, 不用管device
            
            return [loss(要求是标量), model_output]
        '''
        raise NotImplementedError()
    ###############  定义数据如何输入模型，如何得到loss  ###############




    ###############  定义log记什么 计算什么指标  ###############
    def init_log_cache(self) -> Dict[str, list]:
        '''
            初始化 log 所需的空间
        '''
        log_cache = {
            "loss": []
        }
        return log_cache

    def update_log_cache(self, log_cache:dict, data, loss, model_output):
        '''
            根据模型的计算结果更新 log cache
            要求全部内存保存，不能占用显卡空间

            Args:
                log_cache:dict 把数据更新到什么地方去
                data collate_fn 返回的数据类型
                loss 损失
                model_output 自己定义的模型的返回值
        '''
        log_cache['loss'].append(loss.item())
        

    def calculate_matrics_and_save_log(self, log_cache, tag:str):
        '''
            如何利用 log cache 中的信息记录log
            记录后清空 log cache
            返回值为 最终性能评估指标(越高越好)

            Args:
                log_cache 根据什么计算指标
                tag:str 放在tensor board 里面tag打什么
        
        '''

        dict_loss = {
            'total loss': np.array(log_cache['loss']).mean()
        }
        self.config.tbWriter.add_scalars('loss/'+tag, dict_loss, global_step=self.total_batch)

        if tag == 'train':
            dict_lr = {
                'lr': self.scheduler.get_lr()[0]
            }
            self.config.tbWriter.add_scalars('lr', dict_lr, global_step=self.total_batch)
        
        # 清空 cache
        log_cache_init = self.init_log_cache()
        for key in log_cache.keys():
            log_cache[key] = log_cache_init[key]

        return -dict_loss['total loss']
    
    ###############  定义log记什么 计算什么指标  ###############





    ###############  何时保存 log 模型 进行evaluate  ###############
    def should_log(self):
        '''
            什么情况下记录日志
        '''
        return (self.total_batch) % self.config.log_step == 0 
    def should_evaluate(self):
        '''
            什么情况下在测试集上测试
        '''
        return self.total_batch % self.config.eval_step == 0
        
    def should_save(self, current_metrics):
        '''
            什么情况下保存chackpoint
        '''
        return current_metrics != None and current_metrics > self.best_metices
    ###############  何时保存 log 模型 进行evaluate  ###############

    def get_time_dif(self,start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    

    def to_device(self, params:Tuple[torch.Tensor], config:Config):
        list_params_device = []
        for param in params:
            if type(param) == torch.Tensor:
                list_params_device.append(param.to(config.device))
            else:
                list_params_device.append(param)
        return list_params_device
        

    def train(self):
        self.model.train()
        train_log_cache = self.init_log_cache()

        self.start_time = time.time()
        stop = False
        load_data_start = time.perf_counter()
        for epoch in range(self.config.num_epochs):
            self.config.logger.info("******************** Epoch: {}/{} ***********************".format(epoch+1, self.config.num_epochs))
            for i, data in enumerate(self.train_dataloader):
                load_data_end = time.perf_counter()
                # print('data时间:%s毫秒' % ((load_data_end - load_data_start)*1000))
                
                data = self.to_device(data, self.config)
                # model forward
                model_start = time.perf_counter()
                loss, model_output = self.one_step(data)
                # model backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                model_end = time.perf_counter()
                # print('model时间:%s毫秒' % ((model_end - model_start)*1000))
                
                # calculate train acc
                save_start = time.perf_counter()
                self.update_log_cache(train_log_cache, data, loss, model_output, 'train')
                if self.should_log():
                    self.config.logger.info("step: {}/{}, Train loss: {:.2f}".format(self.total_batch%len(self.train_dataloader), len(self.train_dataloader), np.array(train_log_cache['loss']).mean()))
                    self.calculate_matrics_and_save_log(train_log_cache, 'train')
                save_end = time.perf_counter()
                # print('save时间:%s毫秒\n' % ((save_end - save_start)*1000))
                
                # model step
                # if self.total_batch % self.config. == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if (self.total_batch+10) < self.config.num_epochs * len(self.train_dataloader): # 防止学习率为0
                    self.scheduler.step()

                # calculate dev/test acc
                if self.total_batch!=0:
                    self.maybe_log_evaluate_save()

                self.total_batch += 1

                if (self.total_batch - self.last_improve) > self.config.require_improvement:
                    self.config.logger.info("No optimization for a long time, auto-stopping...")
                    stop = True
                    break
                load_data_start = time.perf_counter()
                
            if stop:
                break

        self.config.logger.info("******FINISH TRAINING******")

    def maybe_log_evaluate_save(self):
        '''
            是否要 保存、记录log、evaluate
        '''
        current_metrics = None 
        if self.should_evaluate():
            # self.config.logger.info("step: {}/{}, eval......".format(self.total_batch%len(self.train_dataloader), len(self.train_dataloader)))
            self.model.eval()
            # dev
            if self.dev_dataloader != None:
                dev_metrics = self.evaluate(self.dev_dataloader, 'dev')
                self.config.logger.info("step: {}/{}, dev acc: {}".format(self.total_batch%len(self.train_dataloader), len(self.train_dataloader), dev_metrics))
            # test
            self.config.logger.info("*****evaluating*****")
            current_metrics = self.evaluate(self.test_dataloader, 'test')
            # 打印日志
            improve = '[*]' if current_metrics > self.best_metices else ''
            self.last_imporve = self.total_batch
            self.config.logger.info("step: {}/{}, test acc: {}, Time usage: {} {}".format(self.total_batch%len(self.train_dataloader), len(self.train_dataloader), current_metrics, self.get_time_dif(self.start_time), improve))
            self.model.train()

        if self.should_save(current_metrics):
            if current_metrics != None and current_metrics > self.best_metices: # 最佳情况显示最佳提示
                # self.config.logger.info("###BEST###")
                self.best_metices = current_metrics
            self.save_model()

    def save_model(self):
        if type(self.model) == nn.parallel.DataParallel or type(self.model) == nn.parallel.DistributedDataParallel:
            torch.save(self.model.module.state_dict(), self.config.save_path)
        else:
            torch.save(self.model.state_dict(), self.config.save_path)

    @torch.no_grad()
    def evaluate(self, datloader:DataLoader, tag:str) -> float:
        if self.evaluator == None: # 无独立评测流程
            test_log_cache = self.init_log_cache()
            for i, data in enumerate(datloader):
                data = self.to_device(data, self.config)
                loss, model_output = self.one_step(data)
                self.update_log_cache(test_log_cache, data, loss, model_output, tag)
            metrics = self.calculate_matrics_and_save_log(test_log_cache, tag)
            return metrics
        else: # 定制化评测流程
            return self.evaluator.evaluate(datloader, tag, self.total_batch)
