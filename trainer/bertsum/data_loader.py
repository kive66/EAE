import torch
from torch.utils.data import DataLoader, Dataset
from configs.config import Config
from transformers import BatchEncoding

class BertSumLoader():
    def __init__(self, config):
        self.config = config


    def create_dataset(self, dataset_type):
        dataset = SumDataset(self.config, dataset_type)
        return dataset

    def create_dataloader(self, dataset_type='train', shuffle = False):
        return DataLoader(
            dataset = self.create_dataset(dataset_type),
            batch_size = self.config.batch_size if dataset_type == 'train' else 1,
            shuffle = shuffle if dataset_type == 'train' else False,
            # drop_last = self.config.drop_last if dataset_type == 'train' else False,
            collate_fn = lambda data: self.collate_fn_events(data, dataset_type, self.config),
            num_workers = 1
        )
    
    @staticmethod
    def collate_fn_events(data:list, dataset_type, config:Config):
        pre_src = []
        pre_labels = []
        pre_segs = []
        pre_clss = []
        src_str = []
        tgt_str = []
        pre_sum = []
        
        for ex in data:
            # src = ex['src']
            # sent_labels = ex['labels']
            # src_subtokens = config.tokenizer.tokenize(src)
            # src_subtokens = src_subtokens[:510]
            # src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
            # src_subtoken_idxs = config.tokenizer.convert_tokens_to_ids(src_subtokens)
            # _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == config.tokenizer.vocab['[SEP]']]
            # segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            # segments_ids = []#mask
            # for i, s in enumerate(segs):
            #     if (i % 2 == 0):
            #         segments_ids += s * [0]
            #     else:
            #         segments_ids += s * [1]
            # cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == config.tokenizer.vocab['[CLS]']]
            # sent_labels = sent_labels[:len(cls_ids)] 
            
            # pre_src.append(src_subtoken_idxs)
            # pre_labels.append(sent_labels)
            # pre_segs.append(segments_ids)
            # pre_clss.append(cls_ids)
            pre_src.append(ex['src'])
            pre_labels.append(ex['labels'])
            pre_segs.append(ex['segs'])
            pre_clss.append(ex['clss'])
            src_str.append(ex['src_txt'])
            tgt_str.append(ex['tgt_txt'])
            pre_sum.append(ex['sum_txt'])
            
        def _pad(data, pad_id, width=-1):
            if (width == -1):
                width = max(len(d) for d in data)
            rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
            return rtn_data
        src = torch.tensor(_pad(pre_src, 0))
        labels = torch.tensor(_pad(pre_labels, 0))
        segs = torch.tensor(_pad(pre_segs, 0))
        mask = torch.ones(src.shape)
        mask[src == 0] = 0

        clss = torch.tensor(_pad(pre_clss, -1))
        mask_cls = torch.ones(clss.shape)
        mask_cls[clss == -1] = 0
        clss[clss == -1] = 0
        # summarization = torch.tensor(_pad(pre_sum, 0))
        
        summarization:BatchEncoding = config.tokenizer(
            pre_sum,
            padding="longest",
            max_length=config.max_seq_len,
            truncation=True,
            # return_token_type_ids=True,
            # return_offsets_mapping = True,
            return_tensors="pt"
        )
        return src, labels, segs, clss, mask, mask_cls, src_str, tgt_str, summarization
            

class SumDataset(Dataset):
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
        dataset = torch.load(path)
        return dataset
    def __len__(self):
        return len(self.list_datas)
    
    def __getitem__(self, idx):
        return self.list_datas[idx]
