import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def write_json(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii =False)
        