import json
from numpy import *
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    dataset = ['train', 'val', 'test']
    for type in dataset:
        doc_len = []
        sent_len = []
        event_num = []
        entity_num = []
        data = load_json('add_entity_rams/'+ type + '.json')
        for doc in data:
            doc_key = doc['doc_key']
            evt_triggers = doc['evt_triggers']
            sentences = doc['sentences']
            entities = doc['pred_entity']
            text = []
            for sent in sentences:
                text.extend(sent)
            doc_len.append(len(text))
            sent_len.append([len(sent) for sent in sentences])
            event_num.append(len(evt_triggers))
            entity_num.append(len(doc['pred_entity']))
            
        print(type)
        print(len(data))
        print(max(doc_len))
        print(mean(doc_len))
        doc_sent_len = [max(l) for l in sent_len]
        print(mean(doc_sent_len))
        print(max(event_num))
        print(mean(event_num))
        print(max(entity_num))
        print(mean(entity_num))
        