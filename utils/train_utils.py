import time
from configs.config import Config
from typing import Tuple
import torch
from datetime import timedelta
import random
import numpy as np


def to_device(params:Tuple[torch.Tensor], config:Config):
    list_params_device = []
    for param in params:
        if type(param) == torch.Tensor:
            list_params_device.append(param.to(config.device))
        else:
            list_params_device.append(param)
    return list_params_device



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

