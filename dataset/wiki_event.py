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

class WikiEvent():
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
            {
                doc_id: document_id, 
                tokens: [ words, ... ],
                text: text,
                sentences , 
                entity_mentions: [
                    {
                        id,
                        sent_idx,
                        start,
                        end,
                        entity_type,
                        mention_type,
                        text
                    },
                    ...
                ],
                relation_mentions: [
                    ...
                ],
                event_mentions: [
                    {
                        id,
                        event_type,
                        trigger: {
                            start,
                            end,
                            text,
                            sent_idx
                        },
                        arguments: [
                            {
                                entity_id,
                                role,
                                text
                            },
                            ...
                        ]
                    }
                ]
            },
            ...
        ]    
        '''
        invalid_arg_num = 0
        window_size = self.config.window_size
        list_datas = []
        for line in tqdm(list_dataset):
            # 先拿到entity dict
            entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
            # 只保留包含event的句子
            events = line["event_mentions"]
            # 只保留有触发词的句子
            if not events:
                continue
            doc_id = line['doc_id']
            full_text = line['tokens']
            sent_len = len(full_text)

            for i, event in enumerate(events):
                # 触发词处理
                event_trigger = event['trigger']
                event_type = event['event_type']
                cut_text = full_text
                
                # window切分后的偏移量，起止位置
                offset, min_start, max_end = 0, 0, window_size+1
                if sent_len > window_size + 1:
                    if event_trigger['end'] <= window_size // 2:
                        cut_text = full_text[ : ( window_size + 1 )] # 触发词在左边，往左往右拿window size个字符
                    elif event_trigger['start'] >= (sent_len - window_size//2):
                        offset = sent_len - (window_size+1)
                        # 触发词在右边，往右边切分。左边offset个字符被扔掉
                        min_start += offset
                        max_end += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset
                        cut_text = full_text[-(window_size+1):] # 从右往左拿window size个字符
                    else:
                        offset = event_trigger['start'] - window_size //2
                        min_start += offset
                        max_end += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset
                        cut_text = full_text[offset: (offset + window_size + 1)]
                event_trigger['offset'] = offset

                # 去寻找包含在触发词位置间的论元信息
                event_arguments = []
                if not event['arguments']:
                    continue
                for argument in event['arguments']:
                    argument_entity = entity_dict[argument['entity_id']]
                    event_argument = {
                        'start': argument_entity['start'],
                        'end': argument_entity['end'],
                        'text': argument['text'],
                        'role': argument['role']
                    }
                    if event_argument['start'] >= min_start and event_argument['end'] <= max_end:
                        event_argument['start'] -= offset
                        event_argument['end'] -= offset
                        event_arguments.append(event_argument)
                    else:
                        invalid_arg_num += 1
                list_datas.append(Event(doc_id, None, cut_text, event_type, event_trigger, event_arguments, full_text, None))

        self.config.logger.info("{} examples collected. {} arguments dropped.".format(len(list_datas), invalid_arg_num))
        return list_datas

from configs.paie.config import PaieConfig
if __name__ == "__main__":
    config = PaieConfig()
    config.loadFromFile('./configs/paie/config.json')
    config.initLogger()
    WikiEvent(config,'train')