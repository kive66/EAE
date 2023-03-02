import signal
from trainer.bertsum.data_loader import BertSumLoader
from trainer.bertsum.bertsum_trainer import DecoderTrainer
from models.bertsum.bertsum_model import Summarizer
from configs.bertsum.config import BertSumConfig
from utils.train_utils import set_seed
from torch.nn.parallel import DataParallel
import traceback
import os
import sys
sys.path.append('./')
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    set_seed(42)

    config = BertSumConfig()
    config.loadFromFile('./configs/bertsum/config_wikievents.json')
    config.initTensorBoard()
    config.initLogger()
    config.copyExpScript()

    model = Summarizer(config)
    model.to(config.device)

    model = DataParallel(model, device_ids=[0])
    try:
        # 加载数据集
        data_processer = BertSumLoader(config)
        config.logger.info("load train set......")
        train_dataset = data_processer.create_dataloader('train')
        # config.logger.info("load dev set......")
        # dev_dataset = data_processer.create_dataloader('dev')
        config.logger.info("load test set......")
        test_dataset = data_processer.create_dataloader('test')

        # 训练模型
        trainer = DecoderTrainer(
            config, model, train_dataloader=train_dataset, test_dataloader=test_dataset)
        trainer.train()
    except Exception as e:
        config.logger.error(traceback.format_exc())
