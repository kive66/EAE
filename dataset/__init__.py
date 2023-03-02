import sys
sys.path.append('./')
import json
from dataset.rams import RAMS
from dataset.event import Event, EventDataset
from dataset.ace_event import AceEvent
from dataset.few_fc import FewFC
from dataset.chFinAnn import ChFinAnn
from dataset.potefin import PoTeFin

def get_event_dataset(path):
    # 直接读取抽取出来的结果
    with open(path,'r',encoding='utf8')as fp:
        datasets = json.load(fp)
    data = [Event(**p) for p in datasets]
    return data