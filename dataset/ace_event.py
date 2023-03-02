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

class AceEvent():
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
        list_dataset = self.load_jsonlines(path)
        '''
        [
            Event: {
                sentence: [
                    text, ...
                ],
                s_start: sentence start index,
                ner: [
                    {
                        start,
                        end,
                        type
                    },
                    ...
                ],
                relation: [ ... ],
                event: [
                    [
                        [
                            trigger_start,
                            event_type
                        ],
                        [
                            role_start,
                            role_end,
                            role
                        ],
                        ...
                    ],
                    ... 
                ],
            }
        ]
        '''
        list_datas = []
        for doc_idx, line in enumerate(list_dataset):
            if not line['event']: # 不包含事件的跳过
                continue
            events = line['event']
            full_text = line['sentence']
            offset = line['s_start']
            text = line['sentence']

            for event_idx, event in enumerate(events):
                event_type = event[0][1]
                event_trigger = {
                    'start': event[0][0] - offset,
                    'end': event[0][0] - offset + 1,
                    'text': " ".join(text[event[0][0] - offset:event[0][0] - offset + 1]),
                    'offset': offset
                }

                # 去寻找包含在触发词位置间的论元信息
                event_arguments = []
                if not event[1:]:
                    continue
                for argument in event[1:]:
                    event_argument = {
                        'start': argument[0] - offset,
                        'end': argument[1] - offset + 1,
                        'text': " ".join(text[argument[0] - offset:argument[1] - offset + 1]),
                        'role': argument[2]
                    }
                    event_arguments.append(event_argument)
                list_datas.append(Event(doc_idx, event_idx, text, event_type, event_trigger, event_arguments, full_text, None))

        self.config.logger.info("{} examples collected.".format(len(list_datas)))
        return list_datas

from configs.paie.config import PaieConfig
if __name__ == "__main__":
    config = PaieConfig()
    config.loadFromFile('./configs/paie/config_ace.json')
    config.initLogger()