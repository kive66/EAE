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

class FewFC():
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
                id,
                content,
                event: [
                {
                    type,
                    mentions: [
                        {
                            word,
                            span,
                            role
                        }
                    ]
                }]
            }
        ]
        '''
        list_datas = []
        for doc_idx, line in enumerate(list_dataset):
            text = line['content']
            for i, event in enumerate(line['events']):
                event_type = event['type']

                # 去寻找包含在触发词位置间的论元信息
                event_arguments = []
                for mention in event['mentions']:
                    if mention['role'] == 'trigger':
                        event_trigger = {
                            'start': mention['span'][0],
                            'end': mention['span'][1],
                            'text': mention['word'],
                            'offset': 0
                        }
                        continue
                
                    event_argument = {
                        'start': mention['span'][0],
                        'end': mention['span'][1],
                        'text': mention['word'],
                        'role': mention['role']
                    }
                    event_arguments.append(event_argument)
                list_datas.append(Event(doc_idx, line['id'], list(text), event_type, event_trigger, event_arguments, list(text), None, event_type_id=self.config.event_to_idx[event_type]))
                # list_datas.append(Event(doc_idx, line['id'], list(text), event_type, None, event_arguments, list(text), None, event_type_id=self.config.event_to_idx[event_type]))

        self.config.logger.info("{} examples collected.".format(len(list_datas)))
        return list_datas

from configs.paie.config import PaieConfig
if __name__ == "__main__":
    config = PaieConfig()
    config.loadFromFile('./configs/paie/config_fewfc.json')
    config.initLogger()
    FewFC(config,'train')