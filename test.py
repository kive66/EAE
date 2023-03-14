#!/usr/bin/python
import re
import numpy as np
from transformers import AutoTokenizer
import torch
# line = "Cats are smarter than dogs"

# matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)

# if matchObj:
#    print( "matchObj.group() : ", matchObj.group())


# l = torch.tensor([[1,0.6,0.4,1], [1,0,0,1]])
# l[l>0.5]= 1
# l = range(1,3)

txt = 'dsf123as2sad'

# print(re.findall(r'[0-9]+|[a-z]+', txt))
# l = [[1,2],[3,4]]
# print(torch.tensor(l))
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True) # 构建tokenizer
t = [['!','place of employment', 'placeofemployment'],['aaaa','place of employment', 'placeofemployment']]
t = tokenizer(
            t,
            padding="max_length",
            max_length=12,
            truncation=True,
            # return_token_type_ids=True,
            # return_offsets_mapping = True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
c =tokenizer.tokenize('!')
print(c)
print(t.tokens(0))
print(t.word_to_tokens(0,2).start)
print(t.word_ids(0))
# print(t)
# t1 = tokenizer.tokenize(t[1])
# print(t1)
# 
# event_path = "data/rams/event_role_multiplicities.txt"

# def load_event_dict(path):
#     event_dict = {}
#     m = 0
#     with open(path,'r',encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             roles = line.split()
#             event_dict[roles[0]] = roles[1:]
#             m = max(m, len(roles[1:]))
#     print(m)
#     return event_dict
    
# load_event_dict(event_path)

# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base", do_lower_case=True) # 构建tokenizer
# t = ['你好吗',' 北京大饭店', '火车王']
# t = tokenizer(
#             t,
#             padding="max_length",
#             max_length=12,
#             truncation=True,
#             # return_token_type_ids=True,
#             # return_offsets_mapping = True,
#             return_tensors="pt"
#         )
# print(t)
# srctxt = ['I like pizza', 'you hate dog']
# print(tokenizer.tokenize(srctxt))

# tokenized : 0:[ , ] 1:[ , ]
# doc_token:  for i in range(offset):