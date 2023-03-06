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
        role_labels = []
        role_spans = []
        summarization = []
        bertsum = []
        entities = []
        entity_spans = []
        char2token = []
        entity2token = []
        
        for ex in data:
            tokens.append(ex['tokens'])
            role_names.append(ex['roles'])
            # role_labels.append(ex['role_labels'])
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
            # return_offsets_mapping = True,
            return_tensors="pt"
        )
        # 生成原始句子到分词句子的词映射
        for i in range(len(data)):
            char2token.append(sent_tokens.word_ids(i))
       
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
        # 生成原始句子到分词句子的实体映射
        for i in range(len(data)):
            entity2token.append(entities.word_ids(i))
            
        #与论元角色共同编码    
        summar_embeddings = []                 
        bertsum_embeddings = []
        
        summar_masks = []
        bertsum_masks = []
        
        role_labels = torch.zeros((len(data),config.max_role_num, config.max_seq_len), dtype=torch.float)
        role_start_labels = []
        role_end_labels = []
        for i, roles in enumerate(role_names):
            # 处理batch i的所有roles
            summar_role_embedding = []
            bertsum_role_embedding = []
            
            summar_role_mask = []
            bertsum_role_mask = []
            
            role_start_label = [] # batch i [role_num, start_dim]
            role_end_label = [] # batch i [role_num, end_dim]
            for j, role in enumerate(roles):
                # 处理一个role的多个span
                role_span = role_spans[i][j]
                role_starts= [] # 一个role的所有论元start
                role_ends = [] # 一个role的所有论元end
                for span in role_span:
                    role_starts.append(span[0])
                    role_ends.append(span[1])
                    role_token_start = sent_tokens.char_to_token(i, span[0])
                    role_token_end = sent_tokens.char_to_token(i, span[1])
                    if role_token_start and role_token_end:
                        for k in range(role_token_start, role_token_start):
                            role_labels[i][j][k] = 1
                role_start_label.append(role_starts)
                role_end_label.append(role_ends)
                # 摘要向量化
                summar_embedding:BatchEncoding = config.tokenizer(
                    summarization[i],
                    role,
                    padding="max_length",
                    max_length=config.max_seq_len,
                    truncation='only_first',
                    # return_token_type_ids=True,
                    # return_offsets_mapping = True,
                    return_tensors="pt"
                )
                summar_role_embedding.extend(summar_embedding.input_ids)
                summar_role_mask.extend(summar_embedding.attention_mask)
                # 摘要向量化
                bertsum_embedding:BatchEncoding = config.tokenizer(
                    bertsum[i],
                    role,
                    padding="max_length",
                    max_length=config.max_seq_len,
                    truncation='only_first',
                    # return_token_type_ids=True,
                    # return_offsets_mapping = True,
                    return_tensors="pt"
                )
                bertsum_role_embedding.extend(bertsum_embedding.input_ids)
                bertsum_role_mask.extend(bertsum_embedding.attention_mask)
                
            role_start_labels.append(role_start_label)
            role_end_labels.append(role_end_label)
                
            summar_embeddings.append(torch.stack(summar_role_embedding))
            bertsum_embeddings.append(torch.stack(bertsum_role_embedding))
            
            summar_masks.append(torch.stack(summar_role_mask))
            bertsum_masks.append(torch.stack(bertsum_role_mask))
            
        summar_embeddings = torch.stack(summar_embeddings).transpose(0,1)
        bertsum_embeddings = torch.stack(bertsum_embeddings).transpose(0,1)
        
        summar_masks = torch.stack(summar_masks).transpose(0,1)
        bertsum_masks = torch.stack(bertsum_masks).transpose(0,1)
        
        role_labels = role_labels.transpose(0,1)
        return sent_tokens.input_ids, entities.input_ids, summar_embeddings, bertsum_embeddings, sent_tokens.attention_mask, entities.attention_mask, summar_masks, bertsum_masks, role_names, role_labels, role_spans, entity_spans, char2token, entity2token
            

class Rams(Dataset):
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
            # role_labels = []#文档文本中真实论元角色向量
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
                    if role == arg_role and arg[1]+1 <self.config.max_seq_len:
                        role_span.append([arg[0],arg[1]+1])
                role_spans.append(role_span)        

            for entity in doc_entities:
                if not len(entity):
                    continue
                if isinstance(entity[0][0],str):
                    entities.extend([entity[0][0]])
                else:
                    entities.append(' '.join(entity[0][0]))
                multi_span = []
                for span in entity:
                    if span[-1]+1 <= self.config.max_seq_len:
                        multi_span.append((span[-2],span[-1]+1))
                entity_span.append(multi_span)
                    
            data.append({'tokens':text, 'roles': roles, 'role_spans': role_spans, 'summarization': summar, 'bertsum': bertsum, 'entities': entities, 'entity_span': entity_span})

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