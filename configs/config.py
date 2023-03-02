from transformers import AutoTokenizer,BertTokenizerFast
import torch
import json
import os
from pathlib import Path
from utils.logger_utils import get_logger
import time
from torch.utils.tensorboard import SummaryWriter

class Config(object): # 基础config
    def __init__(self) -> None:
        # basic param
        self.exp_purpose = None
        self.model = None
        self.dataset = None
        self.train_path = None
        self.test_path = None
        self.exp_path = None
        self.save_path = None
        self.project_path = None
        self.save_scriptList = None

        self.do_train = None
        self.do_test = None
        self.require_improvement = None
        self.num_epochs = None
        self.batch_size = None
        self.test_batch_size = None
        self.max_seq_len = None
        self.eval_step = None
        self.log_step = None
        self.pretrain_path = None
        self.hidden_size = None

        self.logger = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        # 优化器参数
        self.basic_learning_rate = None
        self.encoder_learning_rate = None
        self.rate_warmup_steps = None

        # dataLoader 参数
        self.shuffle = None
        self.drop_last = None
        self.num_workers = None
        

    def loadFromFile(self, path:str):
        with open(path, 'r', encoding='utf-8') as in_:
            dict_config = json.load(in_)
        today=str(time.strftime("%Y-%m-%d", time.localtime()))
        self.startTime = time.strftime("%Y%m%d%H%M%S", time.localtime()) # 训练开始时间
        self.model = dict_config['model']
        self.dataset = dict_config['dataset']
        self.exp_purpose = dict_config['exp_purpose']
        self.train_path = dict_config['train_path']
        self.dev_path = dict_config['dev_path']
        self.test_path = dict_config['test_path']

        self.project_path = dict_config['project_path']
        self.exp_path = dict_config['save_path'] # 整个实验的地址
        self.path = Path(dict_config['save_path']+today+'/'+dict_config['dataset']+'_'+dict_config['model']+'_'+str(self.startTime)+'/') # 实验记录保存地址

        self.save_path = str(self.path.joinpath('best.pth'))  # 模型训练结果
        self.log_path = str(self.path.joinpath('logs.log')) # 日志保存地址
        
        self.tensorBoard_path = str(self.path.joinpath('tensorboard/')) # tensorboard 保存

        self.script_path = str(self.path.joinpath('script/')) # 训练脚本保存地址

        self.save_scriptList = dict_config['save_scriptList'] # 保存哪些文件夹下的脚本

        self.do_train = dict_config['do_train']
        self.do_test = dict_config['do_test']

        self.require_improvement = dict_config['require_improvement']
        

        self.num_epochs = dict_config['num_epochs']
        self.batch_size = dict_config['batch_size']
        self.test_batch_size = dict_config['test_batch_size']
        self.max_seq_len = dict_config['max_seq_len']
        self.eval_step = dict_config['eval_step']
        self.log_step = dict_config['log_step']
        self.pretrain_path = dict_config['pretrain_path']
        self.hidden_size = dict_config['hidden_size']

        

        self.rate_warmup_steps = dict_config['rate_warmup_steps']
        self.basic_learning_rate = dict_config['basic_learning_rate']
        self.encoder_learning_rate = dict_config['encoder_learning_rate']

        self.shuffle = dict_config['shuffle']
        self.drop_last = dict_config['drop_last']
        self.num_workers = dict_config['num_workers']


    def initLogger(self, logger=None):
        if logger == None:
            path = self.path # 实验记录保存地址
            if not path.exists():
                os.makedirs(path)
            self.logger = get_logger(str(self.startTime), self.log_path) # 获取日志器
        else:
            self.logger = logger

        # 打印超参数
        self.logger.info("******HYPER-PARAMETERS******")
        for key in self.__dict__.keys():
            self.logger.info("{}: {}".format(key, self.__dict__[key]))
        self.logger.info("****************************")

    def initTensorBoard(self):
        if not os.path.exists(self.tensorBoard_path):
            os.makedirs(self.tensorBoard_path) # 开辟文件夹
        self.tbWriter = SummaryWriter(self.tensorBoard_path)

    def copyExpScript(self):
        if not os.path.exists(self.script_path):
            os.makedirs(self.script_path) # 开辟文件夹
        # 保存训练使用的脚本
        for file in self.save_scriptList:
            os.system('cp -r {}{} {}'.format(self.project_path, file, self.script_path))



