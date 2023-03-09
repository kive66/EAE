import datetime
import signal
from trainer.decoder.data_loader import DecoderLoader
from trainer.decoder.decoder_trainer import DecoderTrainer
from models.decoder.decoder import Decoder
from configs.decoder.config import DecoderConfig
from trainer.decoder.data_loader import DecoderLoader
from utils.train_utils import set_seed
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.multiprocessing import Process
import torch
import traceback
import os
import sys
sys.path.append('./')
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == '__main__':
    set_seed(42)
    
    
    config = DecoderConfig()
    config.loadFromFile('./configs/decoder/config_rams.json')
    config.initTensorBoard()
    config.initLogger()
    config.copyExpScript()
    
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # config.local_rank = local_rank

    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    # config.device = device

    model = Decoder(config)
    model.load_state_dict(torch.load('exp/2023-03-07/rams_bert_20230307151835/best.pth', map_location='cpu'))

    model.to(config.device)
    # model = DataParallel(model, device_ids=[0, 1])
    # multi-gpu training (should be after apex fp16 initialization)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    try:
        # 加载数据集
        data_processer = DecoderLoader(config)
        config.logger.info("load train set......")
        train_dataset = data_processer.create_dataset('train')
        train_sampler = data_processer.create_sampler(train_dataset)
        train_dataloader = data_processer.create_dataloader(train_dataset, train_sampler,  'train')
        # config.logger.info("load dev set......")
        # dev_dataset = data_processer.create_dataloader('dev')
        config.logger.info("load test set......")
        test_dataset = data_processer.create_dataset('test')
        test_sampler = data_processer.create_sampler(test_dataset)
        test_dataloader = data_processer.create_dataloader(test_dataset, test_sampler,  'test')

        # 训练模型
        trainer = DecoderTrainer(
            config, model, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        trainer.train()
    except Exception as e:
        config.logger.error(traceback.format_exc())
