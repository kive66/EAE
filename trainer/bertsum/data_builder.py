import gc
import jsonlines
import json
import os
import argparse
import torch
import logging
from transformers import AutoTokenizer

logger = logging.getLogger()

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True) # 构建tokenizer
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    
    def wiki_preprocess(self, doc_id, tokens, text, sentences, entity_mentions, event_mentions, summarization):

        original_src_txt = [s[-1] for s in sentences]
        key_sent = set()
        for event in event_mentions:
            trigger = event['trigger']
            arguments = event['arguments']
            key_sent.add(trigger['sent_idx'])
            for arg in arguments:
                for entity in entity_mentions:
                    if entity['id'] == arg['entity_id']:
                        key_sent.add(entity['sent_idx'])
            
        key_sent = list(key_sent)
        sent_labels = [0] * len(sentences)
        for i in key_sent:
            sent_labels[i] =1
        
        src_txt = [sent[-1] for sent in sentences]

        # 百分比截断
        src_subtokens = []
        tokenized_sent= []
        doc_len = 0
        doc_len_list = []
        for txt in src_txt:
            tokenized = self.tokenizer.tokenize(txt)
            tokenized_sent.append(tokenized)
            doc_len += (len(tokenized)+2)
            doc_len_list.append(len(tokenized)+2)
        rate = doc_len / 512
        
        # 过滤太短的句子
        idxs = [i for i, l in enumerate(doc_len_list) if l >= 2*rate]
        src_txt = [src_txt[i] for i in idxs]
        sent_labels = [sent_labels[i] for i in idxs]
        tokenized_sent = [tokenized_sent[i] for i in idxs]
        
        for tokenized in tokenized_sent:
            split = int(((len(tokenized)+2)//rate) - 2)
            src_subtokens += ['[CLS]'] + tokenized[:split] + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []#mask
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        if len(sent_labels) != len(cls_ids):
            print('w')
        # token_labels = token_labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in [sentences[i][-1] for i in key_sent]])
        # sum_txt = self.tokenizer.encode(summarization)
        sum_txt = summarization
        return src_subtoken_idxs, sent_labels, segments_ids, cls_ids, src_txt, tgt_txt, sum_txt
    
    def rams_preprocess(self, doc_key, evt_triggers, sentences, ent_spans, gold_evt_links, summarization):
        original_src_txt = [' '.join(s) for s in sentences]
        key_sent = set()
        sent_len_list = []
        sent_len = 0
        for sent in sentences:
            sent_len_list.append([sent_len,sent_len+len(sent)])
            sent_len+=len(sent)
        for trigger in evt_triggers:
             pos = trigger[0]
             for i, l in enumerate(sent_len_list):
                 if pos > l[0] and pos <l[1]:
                     key_sent.add(i)
                     break   
        for ent in ent_spans:
             pos = ent[0]
             for i, l in enumerate(sent_len_list):
                 if pos > l[0] and pos <l[1]:
                     key_sent.add(i)
                     break
        key_sent = list(key_sent)
        sent_labels = [0] * len(sentences)
        for i in key_sent:
            sent_labels[i] =1
        
        src_txt = sentences


        src_subtokens = []
        src_txt = [' '.join(sent) for sent in src_txt]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []#mask
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)] 
        
        # sum_txt = self.tokenizer.encode(summarization)
        sum_txt = summarization
        tgt_txt = '<q>'.join([' '.join(tt) for tt in [sentences[i] for i in key_sent]])
        return src_subtoken_idxs, sent_labels, segments_ids, cls_ids, src_txt, tgt_txt, sum_txt

     
def _format_rams_to_bert(args, mode):
    json_file = args.raw_path + mode + '.json'
    save_file = args.save_path + mode +'.bert.pt'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        # return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    count=0
    for doc in jobs:
        count+=1
        logger.info('doc %d' % count)
        doc_key = doc['doc_key']
        evt_triggers = doc['evt_triggers']
        sentences = doc['sentences']
        ent_spans = doc['ent_spans']
        gold_evt_links = doc['gold_evt_links']
        summarization = doc['summarization_text']
        
        b_data = bert.rams_preprocess(doc_key, evt_triggers, sentences, ent_spans, gold_evt_links, summarization)
        if (b_data is None):
            continue
        indexed_tokens, sent_labels, segments_ids, cls_ids, src_txt, tgt_txt, sum_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "sum_txt": sum_txt}
        datasets.append(b_data_dict)
    logger.info(mode +': '+ str(len(datasets)))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def _format_wiki_to_bert(args, mode):
    json_file = args.raw_path + mode + '.json'
    save_file = args.save_path + mode +'.bert.pt'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        # return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    # with open("xxxx.jl", "r+", encoding="utf8") as f:
    #     for item in jsonlines.Reader(f):
    #         print(item)
    jobs = json.load(open(json_file))
    datasets = []
    count=0
    for doc in jobs:
        count+=1
        logger.info('doc %d' % count)
        doc_id = doc['doc_id']
        tokens = doc['tokens']
        text = doc['text']
        sentences = doc['sentences']
        entity_mentions = doc['entity_mentions']
        event_mentions = doc['event_mentions']
        summarization = doc['summarization_text']
        
        b_data = bert.wiki_preprocess(doc_id, tokens, text, sentences, entity_mentions, event_mentions, summarization)
        if (b_data is None):
            continue
        indexed_tokens, sent_labels, segments_ids, cls_ids, src_txt, tgt_txt, sum_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "sum_txt": sum_txt}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_path", default='data/wikievents/add_sum/')
    parser.add_argument("-save_path", default='data/wikievents/add_sum/')

    parser.add_argument('-n_cpus', default=1, type=int)
    args = parser.parse_args()
    _format_wiki_to_bert(args, mode='train')
    _format_wiki_to_bert(args, mode='dev')
    _format_wiki_to_bert(args, mode='test')
    
    # _format_rams_to_bert(args, mode='train')
    # _format_rams_to_bert(args, mode='dev')
    # _format_rams_to_bert(args, mode='test')
