import json
from numpy import *
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    dataset = ['train', 'dev', 'test']
    count = []
    for type in dataset:
        doc_len = []
        sent_len = []
        event_num = []
        entity_num = []
        data = load_json('data/rams/decoder/'+ type + '.json')
        for doc in data:
            doc_key = doc['doc_key']
            evt_triggers = doc['evt_triggers']
            sentences = doc['sentences']
            entities = doc['pred_entity']
            args = doc['ent_spans']
            text = []
            for sent in sentences:
                text.extend(sent)
            spans = []
            for arg in doc['ent_spans']:
                span = [arg[0], arg[1]]
                if span in spans:
                    count.append(doc_key)
                    break
                else:
                    spans.append(span)
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
    
    with open('count.json', 'w', encoding='utf-8') as f:
        json.dump(count,f,ensure_ascii=False)
        