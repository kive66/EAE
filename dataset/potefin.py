import sys
sys.path.append('./')
import json
from torch.utils.data import Dataset
from configs.config import Config
from tqdm import tqdm
import copy     
import jsonlines
from itertools import chain
from dataset.event import Event

class PoTeFin():
    def __init__(self, config:Config, dataset_type: str) -> None:
        super().__init__()
        self.config = config
        self.list_datas = self.load_dataset(dataset_type)

    def get_path(self,dataset_type:str):
        if dataset_type == 'train':
            path = self.config.train_path
        elif dataset_type == 'dev':
            path = self.config.dev_path
        elif dataset_type == 'test':
            path = self.config.test_path
        else:
            raise Exception("dataset key error")
        return path

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as in_:
            data = json.load(in_)
        return data

    def load_jsonlines(self, path):
        lines = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                lines.append(obj)
        return lines
    
    def load_dataset(self,dataset_type)->list:
        '''
        return:
        [
            Event: {
                doc_id,
                sent_id: None,
                sent, # 经过window切分后的文本
                event_type,
                tirgger: {
                    start,
                    end,
                    text,
                    offeset
                },
                arguments: [
                    {
                        start,
                        end,
                        text,
                        role
                    }
                ],
                full_text, # 源文本
                first_word_locs: None
            }
        ]
        '''
        path = self.get_path(dataset_type)
        list_dataset = self.load_json(path)

        list_datas = []
        window_size = self.config.window_size
        invalid_arg_num = 0
        for doc_idx, line in enumerate(list_dataset):
            id = line['id']
            full_text = list(line['content'])
            sent_len = len(full_text)
            cut_text = full_text
            event = line['events']
            event_type = event['type']

            # window切分后的偏移量，起止位置
            offset, min_start, max_end = 0, 0, window_size+1
            trigger_start, trigger_end = event['trigger']['span'][0], event['trigger']['span'][1]
            if sent_len > window_size + 1:
                if trigger_end <= window_size // 2:
                    cut_text = full_text[ : ( window_size + 1 )] # 触发词在左边，往左往右拿window size个字符
                elif trigger_start >= (sent_len - window_size//2):
                    offset = sent_len - (window_size+1)
                    # 触发词在右边，往右边切分。左边offset个字符被扔掉
                    min_start += offset
                    max_end += offset
                    trigger_start -= offset
                    trigger_end -= offset
                    cut_text = full_text[-(window_size+1):] # 从右往左拿window size个字符
                else:
                    offset = trigger_start - window_size //2
                    min_start += offset
                    max_end += offset
                    trigger_start -= offset
                    trigger_end -= offset
                    cut_text = full_text[offset: (offset + window_size + 1)]

            event_trigger = {
                'start': trigger_start,
                'end': trigger_end,
                'text': event['trigger']['word'],
                'offset': offset
            }
            # 去寻找包含在触发词位置间的论元信息
            event_arguments = []
            for mention in event['mention']:
                if not mention['role'] or not mention['role'].startswith(event_type):
                    continue
                
                event_argument = {
                    'start': mention['span'][0],
                    'end': mention['span'][1],
                    'text': mention['word'],
                    'role': mention['role'].replace(event_type+'_',"")
                }
                if event_argument['start'] >= min_start and event_argument['end'] <= max_end:
                    event_argument['start'] -= offset
                    event_argument['end'] -= offset
                    event_arguments.append(event_argument)
                else:
                    invalid_arg_num += 1
                    # print(event_trigger, event_argument['start'],min_start , event_argument['end'], max_end, offset, sent_len)
                # event_arguments.append(event_argument)
            list_datas.append(Event(doc_idx, id, cut_text, event_type, event_trigger, event_arguments, full_text, None, event_type_id=self.config.event_to_idx[event_type]))
        self.config.logger.info("{} examples collected. {} arguments dropped.".format(len(list_datas), invalid_arg_num))
        return list_datas

from configs.chinese.config import ChineseConfig
if __name__ == "__main__":
    config = ChineseConfig()
    config.loadFromFile('./configs/chinese/config_potefin.json')
    config.initLogger()
    PoTeFin(config,'train')