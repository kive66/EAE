from configs.config import Config
import json
from transformers import AutoTokenizer

class BertSumConfig(Config):
    def __init__(self) -> None:
        super(BertSumConfig, self).__init__()

    def loadFromFile(self, path:str):
        super(BertSumConfig, self).loadFromFile(path)

        with open(path, 'r', encoding='utf-8') as in_:
            dict_config = json.load(in_)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_path) # 构建tokenizer

        # 添加model自定义参数
        self.threshold = dict_config['threshold']
        self.dataset = dict_config['dataset']
        self.max_desc_seq_len = dict_config['max_desc_seq_len']

