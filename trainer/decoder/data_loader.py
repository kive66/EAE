import torch
from torch.utils.data import DataLoader, Dataset
from configs.config import Config
from transformers import BatchEncoding
import json
import re

class DecoderLoader():
    def __init__(self, config:Config):
        self.config = config

    def create_dataset(self, dataset_type):
        dataset = Rams(self.config, dataset_type)
        return dataset

    def create_sampler(self, dataset):
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        sampler = torch.utils.data.RandomSampler(dataset)
        return sampler
        

    def create_dataloader(self, dataset, sampler, dataset_type='train'):
        # if local_rank not in [-1, 0]:
        #     torch.distributed.barrier()
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.config.batch_size if dataset_type != 'test' else self.config.test_batch_size,
            # sampler = sampler,
            shuffle = self.config.shuffle if dataset_type != 'test' else False,
            drop_last = self.config.drop_last if dataset_type == 'train' else False,
            collate_fn = lambda data: self.collate_fn_events(data, self.config),
            num_workers = 1
        )
        # if local_rank == 0:
        #     torch.distributed.barrier()
        return dataloader
    
    @staticmethod
    def collate_fn_events(data:list, config:Config):
        tokens = []
        role_names = []
        role_ids = []
        role_mask = []
        role_labels = []
        role_spans = []
        summarization = []
        bertsum = []
        entities = []
        entity_spans = []
        char2token = []
        span2entity = []
        
        for ex in data:
            tokens.append(ex['tokens'])
            role_names.append(ex['roles'])
            role_labels.append(ex['role_labels'])
            role_spans.append(ex['role_spans'])
            summarization.append(ex['summarization'])
            bertsum.append(ex['bertsum'])
            entities.append(ex['entities'])
            entity_spans.append(ex['entity_span'])
        # 句子整体向量化
        sent_tokens:BatchEncoding = config.tokenizer(
            tokens,
            padding="max_length",
            max_length=config.max_seq_len,
            truncation=True,
            is_split_into_words=True,
            # return_token_type_ids=True,
            return_offsets_mapping = True,
            return_tensors="pt"
        )
        # 生成原始句子到分词句子的词映射
        for i, offset in enumerate(sent_tokens['offset_mapping']):
            c2t = []
            start, end =1, 1
            while end < len(offset):
                if offset[end][0]==0 and offset[end][1]==0:
                    break
                if offset[end][1]> offset[end+1][0]:
                    c2t.append([start,end+1])
                    end += 1
                    start = end
                else:
                    end+=1
            # if len(c2t) != len(tokens[i]):
            #     print('wrong')
            char2token.append(c2t)
        # 摘要向量化
        summarization:BatchEncoding = config.tokenizer(
            summarization,
            padding="max_length",
            max_length=config.max_seq_len,
            truncation=True,
            # return_token_type_ids=True,
            # return_offsets_mapping = True,
            return_tensors="pt"
        )
        # 摘要向量化
        bertsum:BatchEncoding = config.tokenizer(
            bertsum,
            padding="max_length",
            max_length=config.max_seq_len,
            truncation=True,
            # return_token_type_ids=True,
            # return_offsets_mapping = True,
            return_tensors="pt"
        )
        # 实体相连的向量化 
        entities:BatchEncoding = config.tokenizer(
            entities,
            padding="max_length",
            max_length=config.max_seq_len,
            truncation=True,
            is_split_into_words=True,
            # return_token_type_ids=True,
            return_offsets_mapping = True,
            return_tensors="pt"
        )
        for i, offset in enumerate(entities['offset_mapping']):
            s2e = [] # entity[i] : char[a : b] : token[c : d]
            c2t =[] # char[i]: token[a : b]
            start, end =1, 1
            entity_span = entity_spans[i]
            # 计算entity向量化后token->char的映射
            while end < len(offset):
                if offset[end][0]==0 and offset[end][1]==0:
                    break
                if offset[end][1]> offset[end+1][0]:
                    c2t.append([start,end+1])
                    end += 1
                    start = end
                else:
                    end+=1
            # 计算entity在char上的分隔
            p = 0
            for entity in entity_span:
                span_len = entity[0][1] - entity[0][0]
                if p+span_len > len(c2t):
                    break
                s2e.append(c2t[p:p+span_len])
                p+=span_len
            span2entity.append(s2e)
                                 
        for roles in role_names:
            tokenized_role:BatchEncoding = config.tokenizer(
                roles,
                padding="max_length",
                max_length=config.max_seq_len,
                truncation=True,
                # return_token_type_ids=True,
                # return_offsets_mapping = True,
                return_tensors="pt"
            )
            role_ids.append(tokenized_role.input_ids)
            role_mask.append(tokenized_role.attention_mask)
        role_ids = torch.stack(role_ids).transpose(0,1)
        role_mask = torch.stack(role_mask).transpose(0,1)
        # role_names = torch.tensor(role_names)
        role_labels = torch.tensor(role_labels, dtype=torch.float).transpose(0,1)
        # entity_span = torch.tensor(entity_span)
        # char2token = torch.tensor(char2token)
        return sent_tokens.input_ids, summarization.input_ids, bertsum.input_ids, entities.input_ids, role_ids, sent_tokens.attention_mask, summarization.attention_mask, bertsum.attention_mask, entities.attention_mask, role_mask, role_names, role_labels, role_spans, entity_spans, char2token, span2entity
            

class Rams(Dataset):
    def __init__(self, config, dataset_type: str) -> None:
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

    def load_event_dict(self, path):
        event_dict = {}
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                roles = line.split()
                roles += ['']*(self.config.max_role_num-len(roles)+1)
                event_dict[roles[0]] = roles[1:]
        return event_dict
            
    
    def load_dataset(self, dataset_type)->list:
       
        path = self.get_path(dataset_type)
        dataset = load_json(path)
        event_role_dict = self.load_event_dict(self.config.event_path)
        data = []
        # if dataset_type == 'train':
        #     dataset = dataset[1700:]
        for doc in dataset:
            doc_key = doc['doc_key']
            evt_triggers = doc['evt_triggers']
            arguments = doc['ent_spans']
            sentences = doc['sentences']
            doc_entities = doc['pred_entity']
            summar = doc['summarization_text']
            bertsum = doc['bertsum']
            text = []#文档文本
            roles = []#论元角色
            role_labels = []#文档文本中真实论元角色向量
            role_spans = []#文档文本中真实论元角色
            entities = [] #
            entity_span = []
            for sent in sentences:
                text.extend(sent)
            for trigger in evt_triggers:
                event_type = trigger[2][0][0]
                roles.extend(event_role_dict[event_type])
            # 构造token对应角色序列
            for role in roles:
                role_label = [0] * self.config.max_seq_len
                role_span = []
                for arg in arguments:
                    arg_role = arg[2][0][0]
                    arg_role = re.findall(r'[0-9]+|[a-z]+', arg_role)[-1]
                    if role == arg_role:
                        for i in range(arg[0], arg[1]+1):
                            if arg[1]+1 <self.config.max_seq_len:
                                role_label[i] = 1
                        role_span.append([arg[0],arg[1]+1])
                
                role_labels.append(role_label)
                role_spans.append(role_span)        

            for entity in doc_entities:
                if not len(entity):
                    continue
                if isinstance(entity[0][0],str):
                    entities.extend([entity[0][0]])
                else:
                    entities.extend(entity[0][0])
                multi_span = []
                for span in entity:
                    multi_span.append((span[-2],span[-1]+1))
                entity_span.append(multi_span)
                    
            data.append({'tokens':text, 'roles': roles, 'role_labels': role_labels, 'role_spans': role_spans, 'summarization': summar, 'bertsum': bertsum, 'entities': entities, 'entity_span': entity_span})

        return data
    def __len__(self):
        return len(self.list_datas)
    
    def __getitem__(self, idx):
        # print(self.list_datas[idx])
        return self.list_datas[idx]

class Wiki(Dataset):
    def __init__(self, config, dataset_type: str) -> None:
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

    def load_dataset(self,dataset_type)->list:
        path = self.get_path(dataset_type)
        dataset = load_json(path)
        data = []
        for doc in dataset:
            doc_id = doc['doc_id']
            event_mentions = doc['event_mentions']
            summarization = doc['summarization_text']
            bertsum = doc['bertsum']
            triggers = []
            arguments = []
            for event in event_mentions:
                triggers.append(event['trigger'][2])
                arguments.append(event['arguments'])
            data.append({'doc_id':doc_id, 'trigger': triggers, 'summarization': summarization, 'bertsum': bertsum})
        return dataset
    def __len__(self):
        return len(self.list_datas)
    
    def __getitem__(self, idx):
        return self.list_datas[idx]
    
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data