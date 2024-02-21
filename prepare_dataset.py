import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


"""
Data Split
"""
def split2ttv(df):
    sent_groups = df.groupby('sentenceID')
    group_labels = list(sent_groups.groups.keys())
    train_idx, test_valid_idx = train_test_split(group_labels, test_size=0.2, random_state=0)
    test_idx, valid_idx = train_test_split(test_valid_idx, test_size=0.5, random_state=0)
    
    # グループのインデックスを使用してデータフレームを再構築
    train_df = pd.concat([sent_groups.get_group(x) for x in train_idx])
    test_df = pd.concat([sent_groups.get_group(x) for x in test_idx])
    valid_df = pd.concat([sent_groups.get_group(x) for x in valid_idx])

    print(f"Data num. Train:{len(train_df)}, Test:{len(test_df)}, Valid:{len(valid_df)}\n")
    return train_df, test_df, valid_df


"""
Dataset
"""
class BertTokenizer():
    def __init__(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    def tokenize(self, wakati, padding_num, max_token):
        # idに変換
        token_num = len(wakati.split(' '))
        assert token_num <= max_token-1, print('Token num should be under 255.')
        ids = np.array(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + wakati.split(' ') + ['[PAD]']*padding_num))
        
        # マスク作成
        attention_mask = np.array([1]*(token_num+padding_num+1))
        attention_mask[1+token_num:] = 0
        
        return ids, attention_mask


def span2seq(spans, token_len, padding_num, lab2id):
    seq = ['N'] * token_len
    
    for i, j, arg in spans:
        i, j = int(i), int(j)
        seq[i:j+1] = [arg]*(j-i+1)
    seq += ['PAD']*padding_num
    #print(seq)
    ids_seq = [lab2id[s] for s in seq]
    return ids_seq


def mk_patial_labels(token_num, pred, orders, lab2id, longest_token_num):
    partial_srl_labels_for_all_iteration = []
    spans = [(pred['word_start'], pred['word_end'], 'V')]
    
    # init
    partial_srl_labels = span2seq(spans, token_num, longest_token_num - token_num, lab2id)
    partial_srl_labels_for_all_iteration.append(partial_srl_labels)

    # それ以降
    for arg_info in orders:
        spans.append(arg_info)
        partial_srl_labels = span2seq(spans, token_num, longest_token_num - token_num, lab2id)
        partial_srl_labels_for_all_iteration.append(partial_srl_labels)
    return partial_srl_labels_for_all_iteration


def mk_target_labels(orders, lab2id, max_token):
    target_start_label, target_end_label, target_srl_label = [], [], []
    for start, end, label in orders:
        target_start_label.append(start)
        target_end_label.append(end)
        target_srl_label.append(lab2id[label])
    target_start_label.append(max_token-1)    # Null 位置はouputの最終次元とする．
    target_end_label.append(-1)             # dummy. -1 だともしもの時エラーが起こせる？
    target_srl_label.append(-1)             # dummy
    return target_start_label, target_end_label, target_srl_label


def mk_arg_indication(orders, lab2id):
    arg_indication = [0]*(len(lab2id)-3)
    for s, e, arg in orders:
        arg_indication[lab2id[arg]] = 1
    
    return arg_indication


def mapping(set, bert_tokenizer, max_token, lab2id):
    sentences, partial_srl_labels, attention_masks, arg_indications = [], [], [], []             # 入力ベクトル等
    target_start_labels, target_end_labels, target_srl_labels = [], [], []  # 正解ラベル
    target_dict_list = []
    longest_token_num = max(set['num_of_tokens'])
    for sent, pred, token_num, orders in zip(set['sentence'], set['predicate'], set['num_of_tokens'], set['label_order_to_give']):
        ids, attention_mask = bert_tokenizer.tokenize(sent, longest_token_num - token_num, max_token)
        sentences.append(ids)
        attention_masks.append(attention_mask)
        partial_srl_labels.append(mk_patial_labels(token_num, pred, orders, lab2id, longest_token_num))
        arg_indications.append(mk_arg_indication(orders, lab2id))
        target_SL, target_EL, target_SRLL = mk_target_labels(orders, lab2id, max_token)
        target_start_labels.append(target_SL), target_end_labels.append(target_EL), target_srl_labels.append(target_SRLL)
        target_dict = {f'{lab}':[] for lab in lab2id.keys()}
        for s, e, srl in zip(target_SL, target_EL, target_SRLL):
            target_dict[f'{srl}'].append([s, e])
        target_dict_list.append(target_dict)
        
    sentences, attention_masks = torch.tensor(sentences, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)
    # iterが先に来るように変形
    partial_srl_labels = torch.tensor(partial_srl_labels, dtype=torch.long).permute(1,0,2)
    target_start_labels = torch.tensor(target_start_labels, dtype=torch.long).permute(1,0)
    target_end_labels = torch.tensor(target_end_labels, dtype=torch.long).permute(1,0)
    target_srl_labels = torch.tensor(target_srl_labels, dtype=torch.long).permute(1,0)
    arg_indications = torch.tensor(arg_indications)
    
    return [sentences, attention_masks, arg_indications, partial_srl_labels, target_start_labels, target_end_labels, target_srl_labels, target_dict_list]


def mk_dataset(df, batch_size, pretrained_model, max_token, lab2id):
    bert_tokenizer = BertTokenizer(pretrained_model)
   
    #意味役割の数でグループを作成
    df['num_of_tokens'] = df['sentence'].map(lambda x: len(x.split(' ')))
    df['labels_per_a_pred'] = df['args'].map(lambda x: len(x))
    df['label_order_to_give'] = df['args'].map(lambda x: sorted([(e['word_start'], e['word_end'], e['argrole']) for e in x], key = lambda k: np.random.random()))
    df = df.sort_values(by='labels_per_a_pred')
    groups = df.groupby('labels_per_a_pred')
    group_labels = groups.groups.keys()
    
    # 意味役割の数毎にミニバッチを作成し統合
    batch_set = []
    for i, group_label in enumerate(group_labels):
        #print(f"Step {i+1}/{len(group_labels)}")
        group_df = groups.get_group(group_label)  # グループデータを一度だけ取得
        group_size = len(group_df)
        batch_indices = range(0, group_size, batch_size)  # バッチごとの開始インデックスを計算
        #print(group_df.iloc[0,:])
        batch_set += [group_df.iloc[start_idx:start_idx+batch_size] for start_idx in batch_indices]

    #return batch_set
    batch_set = [mapping(set, bert_tokenizer, max_token, lab2id) for set in batch_set]
    return batch_set, groups