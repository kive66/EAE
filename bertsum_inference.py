import signal
from trainer.bertsum.bertsum_evaluator import BertSumEvalutor
from trainer.bertsum.data_loader import BertSumLoader
from trainer.bertsum.bertsum_trainer import DecoderTrainer
from models.bertsum.bertsum_model import Summarizer
from configs.bertsum.config import BertSumConfig
from utils.train_utils import set_seed
from torch.nn.parallel import DataParallel
import os
import torch
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    set_seed(42)
    path = "exp/rams/large/"

    config = BertSumConfig()
    config.loadFromFile('./configs/bertsum/config_rams.json')
    config.initTensorBoard()
    config.initLogger()
    config.copyExpScript()
    
    model = Summarizer(config)
    model.load_state_dict(torch.load(path+'best.pth', map_location='cpu'))
    model.to(config.device)

    model = DataParallel(model, device_ids=[0])
    model.eval()
    
    # eval
    # 加载数据集
    data_processer = BertSumLoader(config)
    config.logger.info("load train set......")
    train_dataset = data_processer.create_dataloader('train')
    config.logger.info("load dev set......")
    dev_dataset = data_processer.create_dataloader('dev')
    config.logger.info("load test set......")
    test_dataset = data_processer.create_dataloader('test')


    # 重新构造数据集，将第一阶段事件结果作为第二阶段的训练数据
    evalutor = BertSumEvalutor(config, model)
    # loss, inference_results = evalutor.evaluate(test_dataset,'test')
    # loss, inference_results = evalutor.evaluate(dev_dataset,'dev')
    loss, inference_results = evalutor.evaluate(train_dataset,'train')
    config.logger.info("inference test finished. loss: {}".format(loss))


    dataset_path = "data/rams/add_sum/"
    # 写文件
    with open(dataset_path + 'train.json', 'r', encoding='utf-8') as f:
        with open(path + 'train.json', 'w', encoding = 'utf-8') as w:
            data = json.load(f)
            for i, doc in enumerate(data):
                doc['bertsum'] = inference_results[i]
            w.write(json.dumps(data, ensure_ascii=False))
    config.logger.info("writing test dataset finished.")
            
    
    
    
    
