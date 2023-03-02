from configs.config import Config
from torch.utils.data import DataLoader
import torch.nn as nn

class EvaluatorBasic():
    def __init__(self, config:Config, model:nn.Module) -> None:
        self.config = config
        self.model = model
        self.model.eval()

    def evaluate(self, dataloader:DataLoader, tag:str, total_step=None) -> float:
        raise NotImplementedError()


    