import sys
import time
from tqdm import tqdm
import os
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig

sys.path.append('../')
from network import SentenceEncoder, SRLEncoder, StartDecoder, EndDecoder, SRLDecoder, GumbelSoftmaxPolicy
from prepare_dataset import split2ttv, mk_dataset
from Evaluation.evaluate import cal_label_f1, cal_span_f1, span2seq, get_pred_dic_lists


MAX_TOKEN = 256     # CLSを含めた次元．データセット内の最大トークンは252
BATCH_SIZE = 32     # 大きい方がよい?
ITERS_TO_ACCUMULATE = 1
MAX_ITER = 7        # データセット内最大 srl 数．
DATAPATH = 'Data/common_data_v3_bert.json'
PRETRAINED_MODEL = "cl-tohoku/bert-base-japanese-v2"
VERSION = 'RL_random'
MODEL_NAME = 'batch32_commonv3'
MODEL_PATH = f"models/{PRETRAINED_MODEL.replace('/','')}/{VERSION}"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def mk_filter_s(srl_labels, output_dim, lab2id):
    seq_len = len(srl_labels)
    filter_s = torch.tensor([True]*(seq_len) + [False]*(output_dim-seq_len), device="cuda:0")
    assert len(filter_s) == output_dim
    
    for i, label_id in enumerate(srl_labels):
        if label_id == lab2id['N']:
            continue
        else:
            filter_s[i] = False
    filter_s[-1] = True
    return filter_s


def mk_filter_e(srl_labels, output_dim, lab2id, start):
    filter_e = torch.tensor([False]*(output_dim), device="cuda:0")
    
    for i, label_id in enumerate(srl_labels[start:], start=start):
        if label_id == lab2id['N']:
            filter_e[i] = True
        else:
            break
    
    return filter_e


def mk_filter_srl(srl_left_indication):
    filter_srl = srl_left_indication >= 1
    
    return filter_srl



if __name__ == '__main__':
    print(f'MAX_TOKEN = {MAX_TOKEN}, BATCH_SIZE = {BATCH_SIZE}, VERSION = {VERSION}\n\n')
    with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
        df = pd.read_json(json_file)

    # ラベル辞書作成
    arg_list=[]
    for args in df['args']:
        for arg in args: 
            arg_list.append(arg['argrole'])
    labels = set(arg_list+['N', 'V', 'PAD'])
    labels = sorted(list(labels))
    lab2id = dict( list(zip(labels, range(len(labels)))) )
    id2lab = {v:k for k, v in lab2id.items()}
    #print(lab2id)

    # 各種データ作成（学習，テスト，検証
    train_df, test_df, valid_df = split2ttv(df)
    
    # テスト時のデータ削減（本実験ではコメントアウト）
    train_df = train_df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    valid_df = valid_df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    test_df = test_df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    
    """ Dataset """
    valid_dataset, _ = mk_dataset(valid_df, 1, PRETRAINED_MODEL, MAX_TOKEN, lab2id)
    test_dataset, groups = mk_dataset(test_df, 1, PRETRAINED_MODEL, MAX_TOKEN, lab2id)
    
    """Model"""
    # 各種定義
    loss_function = nn.NLLLoss()    # 損失関数の設定
    device = torch.device("cuda:0")  # GPUの設定

    # モデル定義
    srl_config = BertConfig(
        num_hidden_layers=3,     # Transformerのレイヤー数
        hidden_size=256,         # 768 の 1/3 
        num_attention_heads=4    # 1つのヘッドが64になるように設定
    )
    decoder_config1 = BertConfig(
        num_hidden_layers=1,     # Transformerのレイヤー数
        hidden_size=768+256,     # token + srl
        num_attention_heads=16   # 1つのヘッドが64になるように設定
    )
    decoder_config2 = BertConfig(
        num_hidden_layers=1,     # Transformerのレイヤー数
        hidden_size=768*2+256,   # token + srl + token_s
        num_attention_heads=28   # 1つのヘッドが64になるように設定
    )
    decoder_config3 = BertConfig(
        num_hidden_layers=2,     # Transformerのレイヤー数
        hidden_size=768*2+256*2, # token_s + srl + token_e + srl
        num_attention_heads=32   # 1つのヘッドが64になるように設定
    )
    sentence_encoder = SentenceEncoder(PRETRAINED_MODEL).to(device)
    srl_encoder = SRLEncoder(srl_config, len(lab2id)).to(device)
    start_decoder = StartDecoder(decoder_config1, MAX_TOKEN).to(device)    # CLSを抜かした255トークンの位置 ＋ null の位置 ＝ 256次元（MAX_TOKEN）
    end_decoder = EndDecoder(decoder_config2, MAX_TOKEN-1).to(device)
    srl_decoder = SRLDecoder(decoder_config3, len(lab2id)-3).to(device)    # len(lab2id)-3 なのは，N, V，PAD が 本物のSRLラベルではないから
    policy = GumbelSoftmaxPolicy(len(lab2id)-3, 768+256, MAX_TOKEN-1).to(device)

    sentence_encoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SentenceEncoder.pth'))
    srl_encoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SrlEncoder.pth'))
    start_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_StartDecoder.pth'))
    end_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_EndDecoder.pth'))
    srl_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SrlDecoder.pth'))
    
    """Train"""
    # まずは全部OFF
    for param in sentence_encoder.parameters():
        param.requires_grad = False

    # BERTの最終4層とクラス分類のパラメータを有効化
    layers_to_unfreeze = 4
    for layer in sentence_encoder.bert.encoder.layer[-layers_to_unfreeze:]: # -4 -> 8という意味
        for name, param in layer.named_parameters():
            param.requires_grad = True
            #print(name)

    # 事前学習済み層とクラス分類層の学習率を設定
    optimizer2 = optim.SGD(policy.parameters(), lr=0.01)
    
    print(f'layers_to_unfreeze = {layers_to_unfreeze}')
    print(f'Rate : 5e-5, 1e-4, 1e-4, 1e-4, 5e-5')
    

    start_time = time.time()
    patience = 0
    prev_epoch_policy_loss = 0
    for epoch in range(100):
        epoch_policy_loss = 0
        policy.train(), sentence_encoder.eval(), srl_encoder.eval(), start_decoder.eval(), end_decoder.eval(), srl_decoder.eval()
        train_dataset, _ = mk_dataset(train_df, BATCH_SIZE, PRETRAINED_MODEL, MAX_TOKEN, lab2id, epoch)    # random にするためここで作成
        
        for batch_ids, batch_attention_masks, batch_arg_indications, iter_batch_srl_labels, _, _, _, target_dicts in train_dataset:
            # gpuへ転送
            batch_ids, batch_attention_masks = batch_ids.to(device), batch_attention_masks.to(device)
            batch_arg_indications, iter_batch_srl_labels = batch_arg_indications.to(device), iter_batch_srl_labels.to(device)

            # making data

            with torch.no_grad():
                token_embs = sentence_encoder(batch_ids, batch_attention_masks)                     # batch, token, hidden
            
            batch_loss_sum, policy_loss_sum = 0, 0
            for i, (batch_srl_labels) in enumerate(iter_batch_srl_labels):     # e, srl はNoneで埋め合わされる
                
                iter_size, batch_size, token_len = iter_batch_srl_labels.shape
                
                # 反復試行の最終でないなら
                if i != iter_size-1:
                    with torch.no_grad():
                        srl_embs = srl_encoder(batch_srl_labels, batch_attention_masks[:,1:])       # [batch, token, hidden]
                    
                    # get next transition
                    probs_rl, next_trans = policy(batch_arg_indications, torch.concatenate([token_embs, srl_embs], dim=2))    # SRLモデルに影響を与えないようにdetach
                    
                    with torch.no_grad():
                        # start
                        prob_s_rl = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        for j, srl_labels in enumerate(batch_srl_labels):
                            filter_s = mk_filter_s(srl_labels, MAX_TOKEN, lab2id)
                            prob_s_rl[j] = torch.where(filter_s.unsqueeze(0), prob_s_rl[j], torch.tensor(float('-inf'), device="cuda:0"))
                        starts = torch.argmax(prob_s_rl, dim=1)

                        # starts 内のid がNullを示す255の場合エラーが起きる → starts自体は変更せず，一時的に0をトークンエンベディングとする
                        start_tokens = token_embs[torch.arange(batch_size), torch.where(starts==255, torch.tensor(0, device="cuda:0"), starts)].unsqueeze(1) # 
                        start_tokens_expanded = start_tokens.expand(-1, token_len, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                        
                        # end
                        prob_e_rl = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                        for j, srl_labels in enumerate(batch_srl_labels):
                            filter_e = mk_filter_e(srl_labels, MAX_TOKEN-1, lab2id, starts[j])
                            prob_e_rl[j] = torch.where(filter_e.unsqueeze(0), prob_e_rl[j], torch.tensor(float('-inf'), device="cuda:0"))
                        ends = torch.argmax(prob_e_rl, dim=1)
                        end_tokens = token_embs[torch.arange(batch_size), ends].unsqueeze(1) # [batch, 1, hidden]
                        
                        # srl
                        # starts 内のid がNullを示す255の場合エラーが起きる → starts自体は変更せず，一時的に0をトークンエンベディングとする
                        start_end_vecs = torch.concatenate([start_tokens, srl_embs[torch.arange(batch_size), torch.where(starts==255, torch.tensor(0, device="cuda:0"), starts)].unsqueeze(1),
                                                            end_tokens, srl_embs[torch.arange(batch_size), ends].unsqueeze(1)], dim=2)
                        prob_srl_rl = srl_decoder(start_end_vecs)
                        srl = torch.argmax(prob_srl_rl, dim=1)
                        
                        # 正解データ作成
                        targets = []
                        for j in range(batch_size):
                            try:
                                targets.append(target_dicts[j][next_trans[j]].pop())
                            except:
                                targets.append([MAX_TOKEN-1,-1,-1])
                        preds = list(zip(starts, ends, srl))
                        rewards = policy.cal_rewards(preds, targets, MAX_TOKEN)
                    
                    # Backward
                    batch_policy_loss = torch.tensor(0, device='cuda:0', dtype=torch.float)
                    for reward, probs, trans in zip(rewards, probs_rl, next_trans):
                        batch_policy_loss += -(reward * torch.log(probs[trans]) ) / batch_size
                    batch_policy_loss.backward(retain_graph=True)
                    policy_loss_sum += batch_policy_loss.item()
                
                # 反復試行の最終であれば
                else:
                    with torch.no_grad():   # 計算グラフを保持しつつ，以下の箇所では記録を行わない
                        srl_embs = srl_encoder(batch_srl_labels, batch_attention_masks[:,1:])       # [batch, token, hidden]
                    
                    # get next transition
                    probs_rl, next_trans = policy(batch_arg_indications, torch.concatenate([token_embs, srl_embs], dim=2))
                    
                    with torch.no_grad():   # 計算グラフを保持しつつ，以下の箇所では記録を行わない
                        # start
                        prob_s_rl = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        for j, srl_labels in enumerate(batch_srl_labels):
                            filter_s = mk_filter_s(srl_labels, MAX_TOKEN, lab2id)
                            prob_s_rl[j] = torch.where(filter_s.unsqueeze(0), prob_s_rl[j], torch.tensor(float('-inf'), device="cuda:0"))
                        starts = torch.argmax(prob_s_rl, dim=1)
                    
                    targets = [[MAX_TOKEN-1]*3]*batch_size
                    preds = list(zip(starts, [-1]*batch_size, [-1]*batch_size))
                    rewards = policy.cal_rewards(preds, targets, MAX_TOKEN)
                    
                    # backprop
                    batch_policy_loss = 0
                    for reward, probs, trans in zip(rewards, probs_rl, next_trans):
                        batch_policy_loss += (reward * torch.log(probs[trans])) / batch_size / ITERS_TO_ACCUMULATE

                    batch_policy_loss.backward(retain_graph=True)
                    policy_loss_sum += batch_policy_loss.item()
                    epoch_policy_loss += policy_loss_sum

            if (i + 1) % ITERS_TO_ACCUMULATE == 0:
                optimizer2.step()
                policy.zero_grad()
            if epoch_policy_loss > prev_epoch_policy_loss:
                prev_epoch_policy_loss = epoch_policy_loss
                torch.save(policy.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_policy.pth')
        print(f"############## epoch {epoch} \t, rl_loss {epoch_policy_loss} ##############\n")

    print('Train Time = ', (time.time() - start_time) / 60)
        