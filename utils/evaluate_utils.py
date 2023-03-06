from typing import DefaultDict
from utils.json_utils import load_json
import torch

def calculate(config, pred_label_idx, true_label_idx, slot_types):
    docs = load_json(config.test_path)
    # 首先，根据slot_type分组
    true_positive_c = 0
    true_positive_i = 0
    pred_positive_c, gold_positive_c = 0, 0
    pred_positive_i, gold_positive_i = 0, 0
    for i in range(len(pred_label_idx)):
        sentences = docs[i]['sentences']
        text = []
        for sent in sentences:
            text.extend(sent)
        slot_type = [slot for slot in slot_types[i] if slot != '']
        slot_len = len(slot_type)
        # 去掉不计算的slot指标（padding的）
        cur_gold_label_idx = true_label_idx[i][:slot_len]
        cur_pred_label_idx = pred_label_idx[i][:slot_len]
        gold_ai_data, gold_ac_data = gen_tuples(text, slot_type, cur_gold_label_idx)
        pred_ai_data, pred_ac_data = gen_tuples(text, slot_type, cur_pred_label_idx)
        
        pred_positive_c += len(pred_ac_data)
        gold_positive_c += len(gold_ac_data)
        true_positive_c += count_tp(gold_ac_data, pred_ac_data)
        
        pred_positive_i += len(pred_ai_data)
        gold_positive_i += len(gold_ai_data)
        true_positive_i += count_tp(gold_ai_data, pred_ai_data)
    
    p_c,r_c,f_c = metric(true_positive_c, pred_positive_c, gold_positive_c)
    p_i,r_i,f_i = metric(true_positive_i, pred_positive_i, gold_positive_i)
    return {'p_c': p_c, 'r_c':r_c, 'f_c': f_c, 'p_i':p_i, 'r_i': r_i,'f_i':f_i}
    
    
def metric(true_positive_n, pred_mention_n, gold_mention_n):
    '''
        计算p, f, r
    '''
    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_mention_n != 0:
        prec_c = true_positive_n / pred_mention_n
    else:
        prec_c = 0
    if gold_mention_n != 0:
        recall_c = true_positive_n / gold_mention_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0
    return prec_c, recall_c, f1_c
        
        
def count_tp(y_gold, y_pred):
    '''
        计算tp
    '''
    true_positive = 0
    for y in y_pred:
        if y in y_gold:
            true_positive += 1
    
    return true_positive


def gen_tuples(text, roles, data):
    '''
        通过span从原文中获取论元词构造(word, role)的tuple
    '''
    ai_data = set()
    ac_data = set()
    for i, role in enumerate(roles):
        for arg_span in data[i]:
            ai_one = (' '.join(text[arg_span[0]: arg_span[1]]))
            ac_one = (' '.join(text[arg_span[0]: arg_span[1]]), role)
            ai_data.add(ai_one)
            ac_data.add(ac_one)
    return list(ai_data), list(ac_data)