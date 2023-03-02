class Event:
    def __init__(self, doc_id, sent_id, sent, type, trigger, arguments, full_text, first_word_locs=None, event_type_id = None, **args):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.type = type
        self.trigger = trigger
        self.arguments = arguments
        
        self.full_text = full_text
        self.first_word_locs = first_word_locs
        self.event_type_id = event_type_id
        self.__dict__.update(args) # 导入参数


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = ""
        s += "doc id: {}\n".format(self.doc_id)
        s += "sent id: {}\n".format(self.sent_id)
        s += "text: {}\n".format(" ".join(self.sent))
        s += "event_type: {}\n".format(self.type)
        s += "trigger: {}\n".format(self.trigger['text'])
        for arg in self.args:
            s += "arg {}: {} ({}, {})\n".format(arg['role'], arg['text'], arg['start'], arg['end'])
        s += "----------------------------------------------\n"
        return s

from torch.utils.data import Dataset
class EventDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
