import json
import jsonlines


def read_json(path):
    with open(path,'r',encoding='utf-8')as f:
        data = json.load(f)
    return data

def write_json(result, path):
    with open(path,'w',encoding='utf-8')as f:
        json.dump(result, f, ensure_ascii=False)

type = ['train','dev','test']
for t in type:
    summar = read_json('data/rams/add_sum/'+t+'.json')
    entity = read_json('data/rams/add_entity_rams/'+ t+'.json')

    result = entity
    if len(summar) != len(entity):
        print('wrong')
    for i in range(len(entity)):
        doc_s = summar[i]['doc_key']
        doc_e = entity[i]['doc_key']
        if doc_s != doc_e:
            print('wrong')
            break
        result[i]['bertsum'] = summar[i]['bertsum']
    write_json(result, 'data/rams/decoder/'+t+'.json')
        