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

class ChFinAnn():
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
        '''
        [
            Event: {
                id,
                {
                    sentences,
                    ann_valid_mspans: [entity, entity, ...],
                    ann_valid_dranges: [
                        [
                            sentence_id,
                            start_idx,
                            end_idx
                        ],
                        ...
                    ],
                    ann_mspan2dranges: {
                        entity: [
                            [
                                sentence_id,
                                start_idx,
                                end_idx
                            ],
                            ...
                        ]
                    },
                    ann_mspan2guess_field: {
                        argument: role
                    },
                    recguid_eventname_eventdict_list: [
                        [
                            id,
                            event_type,
                            {
                                role: argument
                            }
                        ],
                        ...
                    ]
                    
                }
            }
        ]
        '''

        def get_sentence_index(sentence, sentence_index, word_index):
            if sentence_index == 0: return word_index
            return sum([len(sentence[i]) for i in range(sentence_index)]) + word_index

        list_datas = []
        for doc_idx, line in enumerate(list_dataset):
            id = line[0]
            text = list("".join(line[1]['sentences']))
            sentence = line[1]['sentences']

            entity_to_index = line[1]['ann_mspan2dranges']
            
            for i, event in enumerate(line[1]['recguid_eventname_eventdict_list']):
                event_type = event[1]
                # 去寻找包含在触发词位置间的论元信息
                event_arguments = []
                for role in event[2].keys():
                    if not event[2][role]:
                        continue
                    argument_entity = entity_to_index[event[2][role]][0]
                    
                    event_argument = {
                        'start': get_sentence_index(sentence, argument_entity[0], argument_entity[1]),
                        'end': get_sentence_index(sentence, argument_entity[0], argument_entity[2]),
                        'text': event[2][role],
                        'role': role
                    }
                    event_arguments.append(event_argument)
                list_datas.append(Event(doc_idx, id, text, event_type, None, event_arguments, sentence, None, event_type_id=self.config.event_to_idx[event_type]))
        self.config.logger.info("{} examples collected.".format(len(list_datas)))
        return list_datas

from configs.chinese.config import ChineseConfig
if __name__ == "__main__":
    config = ChineseConfig()
    config.loadFromFile('./configs/chinese/config_chFinAnn.json')
    config.initLogger()
    ChFinAnn(config,'train')