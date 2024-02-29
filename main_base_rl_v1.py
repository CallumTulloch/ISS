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
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

sys.path.append('../')
from network import SentenceEncoder, SRLEncoder, StartDecoder, EndDecoder, SRLDecoder, GumbelSoftmaxPolicy
from prepare_dataset import split2ttv, mk_dataset, mk_dataset_rl
from Evaluation.evaluate import cal_label_f1, cal_span_f1, span2seq, get_pred_dic_lists


MAX_TOKEN = 256     # CLSを含めた次元．データセット内の最大トークンは252
BATCH_SIZE = 32     # 大きい方がよい?
ITERS_TO_ACCUMULATE = 1
MAX_ITER = 7        # データセット内最大 srl 数．
DATAPATH = 'Data/common_data_v2_bert.json'
DATAPATH = '/data1/takelab/takelab-callum/Desktop/ISS/Data/common_data_v3_bert.json'
PRETRAINED_MODEL = "cl-tohoku/bert-base-japanese-v2"
VERSION = 'RL_random_v1'
MODEL_NAME = 'batch32_commonv3'
TRAIN_FLAG = True
RL_FLAG = True
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
    if all(srl_left_indication==0):
        filter_srl = torch.tensor([False] * (len(srl_left_indication)+1))
        filter_srl[-1] = True  # 終了ラベルのみ
    else:
        filter_srl = srl_left_indication >= 1
        filter_srl = torch.cat([filter_srl, torch.tensor([False])])
    
    return filter_srl


def test(sentence_encoder, srl_encoder, start_decoder, end_decoder, srl_decoder, dataset, lab2id, id2lab):
    predictions, answers = [], []
    sentence_encoder.eval(), srl_encoder.eval(), start_decoder.eval(), end_decoder.eval(), srl_decoder.eval()
    with torch.no_grad():
        for batch_ids, batch_attention_masks, batch_arg_indications, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl, target_dicts in dataset:
            batch_ids, batch_attention_masks = batch_ids.to(device), batch_attention_masks.to(device)
            init_srl_labels = iter_batch_srl_labels[0].to(device)   # testにおいては，iterは最初のVのみでよい．
            iter_target_s, iter_target_e, iter_target_srl = iter_target_s.to(device), iter_target_e.to(device), iter_target_srl.to(device)

            # 正解データ作成. 最後のデータはダミー. SRL評価には必要ない
            answer, prediction = [], []
            for i, (batch_target_s, batch_target_e, batch_target_srl) in enumerate(zip(iter_target_s, iter_target_e, iter_target_srl)):
                for target_s, target_e, target_srl in zip(batch_target_s, batch_target_e, batch_target_srl):
                    if i != len(iter_target_s)-1:
                        answer.append((target_s.item(), target_e.item(), id2lab[target_srl.item()]))
            answers.append(answer)

            # 推測
            token_embs = sentence_encoder(batch_ids, batch_attention_masks)         # batch, token, hidden
            srl_labels = init_srl_labels
            batch_size, token_len, hidden = token_embs.shape
            #print(token_embs.shape, srl_labels.shape)
            
            # start に Null を予測するまではループ
            for iter in range(MAX_ITER):
                srl_embs = srl_encoder(srl_labels, batch_attention_masks[:,1:])     # iなのは1データずつだから
                
                # 開始位置
                prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                filter_s = mk_filter_s(srl_labels[0], MAX_TOKEN, lab2id)
                prob_s = torch.where(filter_s.unsqueeze(0), prob_s, torch.tensor(float('-inf'), device="cuda:0"))
                start = torch.argmax(prob_s, dim=1)[0]
                if start == MAX_TOKEN-1:              # 最終次元はNullを示す．-1 なのはargmaxが理由
                    break
                
                start_tokens = token_embs[torch.arange(batch_size), [start]].unsqueeze(1) # [batch, 1, hidden]
                start_tokens_expanded = start_tokens.expand(-1, token_len, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                
                # 終了位置
                #print(token_embs.shape, srl_embs.shape, start_tokens_expanded.shape)
                prob_e = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                filter_e = mk_filter_e(srl_labels[0], MAX_TOKEN-1, lab2id, start)
                prob_e = torch.where(filter_e.unsqueeze(0), prob_e, torch.tensor(float('-inf'), device="cuda:0"))
                end = torch.argmax(prob_e, dim=1)[0]
                end_tokens = token_embs[torch.arange(batch_size), [end]].unsqueeze(1) # [batch, 1, hidden]
                
                # 意味役割と次回のiter用にsrl_labelsを更新
                start_end_vecs = torch.concatenate([start_tokens, srl_embs[torch.arange(batch_size), [start]].unsqueeze(1),
                                                    end_tokens, srl_embs[torch.arange(batch_size), [end]].unsqueeze(1)], dim=2)
                prob_srl = srl_decoder(start_end_vecs)
                
                srl = torch.argmax(prob_srl, dim=1)
                srl_labels[0][start:end+1] = srl.expand(end-start+1)    # end-start >= 0 は保証されている．
                
                # 正解と予測の作成
                prediction.append((start.item(), end.item(), id2lab[srl.item()]))
            predictions.append(prediction)

        return predictions, answers


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
    # print(lab2id)
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
    start_decoder = StartDecoder(decoder_config1, MAX_TOKEN).to(device)          # CLSを抜かした255トークンの位置 ＋ null の位置 ＝ 256次元（MAX_TOKEN）
    end_decoder = EndDecoder(decoder_config2, MAX_TOKEN-1).to(device)
    srl_decoder = SRLDecoder(decoder_config3, len(lab2id)-3).to(device)    
    policy = GumbelSoftmaxPolicy(len(lab2id)-3+1, 768+256, MAX_TOKEN-1).to(device) # SRLラベル ＋ 終了ラベル

    """Train"""
    if TRAIN_FLAG:
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
        optimizer = optim.AdamW([
            {'params': layer.parameters(), 'lr': 5e-5} for layer in sentence_encoder.bert.encoder.layer[-layers_to_unfreeze:]] + 
           [{'params': srl_encoder.parameters(), 'lr': 1e-4},
            {'params': start_decoder.parameters(), 'lr': 1e-4},
            {'params': end_decoder.parameters(), 'lr': 1e-4},
            {'params': srl_decoder.parameters(), 'lr': 5e-5}]
        )
        optimizer2 = optim.SGD(policy.parameters(), lr=1e-4)
        
        print(f'layers_to_unfreeze = {layers_to_unfreeze}')
        print(f'Rate : 5e-5, 1e-4, 1e-4, 1e-4, 5e-5')
        

        start_time = time.time()
        patience = 0
        prev_f1, prev_start_acc, prev_end_acc, prev_srl_acc = 0, 0, 0, 0
        for epoch in range(100):
            epoch_loss, epoch_policy_loss = 0, 0
            sentence_encoder.train(), srl_encoder.train(), start_decoder.train(), end_decoder.train(), srl_decoder.train()
            train_dataset, _ = mk_dataset_rl(train_df, BATCH_SIZE, PRETRAINED_MODEL, MAX_TOKEN, lab2id, epoch)    # random にするためここで作成
            
            for batch_ids, batch_attention_masks, batch_srl_labels, batch_arg_indications, target_dicts, iter_len in train_dataset:
                batch_ids, batch_attention_masks, batch_srl_labels = batch_ids.to(device), batch_attention_masks.to(device), batch_srl_labels.to(device)
                
                token_embs = sentence_encoder(batch_ids, batch_attention_masks)             # batch, token, hidden

                batch_loss_sum, policy_loss_sum, batch_policy_loss = 0, 0, torch.tensor(0, device=device, dtype=torch.float)
                prev_trans = torch.tensor([0]*(len(lab2id)-3+1))
                batch_size, seq_len = batch_ids.shape
                G = torch.tensor([0]*batch_size, device=device)
                for i in range(iter_len):
                    # 反復試行の最終でないなら
                    if i != iter_len-1:
                        """ RL """
                        srl_embs = srl_encoder(batch_srl_labels, batch_attention_masks[:,1:])       # [batch, token, hidden]

                        # get next transition
                        if i == 0:
                            probs_rl = policy((batch_arg_indications>=1).int().to(device), torch.tensor( [[0]*(len(id2lab)-3)] * batch_size ).to(device), 
                                              torch.concatenate([token_embs.detach(), srl_embs.detach()], dim=2))    # SRLモデルに影響を与えないようにdetach
                        else:
                            probs_rl = policy((batch_arg_indications>=1).int().to(device), F.one_hot(prev_trans, num_classes=len(id2lab)-3).to(device),
                                              torch.concatenate([token_embs.detach(), srl_embs.detach()], dim=2))    # SRLモデルに影響を与えないようにdetach
                            
                        next_trans = torch.tensor([-1]*batch_size)
                        for j, arg_indications in enumerate(batch_arg_indications):
                            filter_srl = mk_filter_srl(arg_indications).to(device)
                            next_trans[j] = torch.argmax(torch.where(filter_srl.unsqueeze(0), probs_rl[j], torch.tensor(float('-inf'), device=device)), dim=1)
                            batch_arg_indications[j, next_trans[j]] -= 1
                        prev_trans = next_trans
                            
                        with torch.no_grad():   # 計算グラフを保持しつつ，以下の箇所では記録を行わない
                            # start
                            prob_s_rl = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                            for j, srl_labels in enumerate(batch_srl_labels):
                                filter_s = mk_filter_s(srl_labels, MAX_TOKEN, lab2id)
                                prob_s_rl[j] = torch.where(filter_s.unsqueeze(0), prob_s_rl[j], torch.tensor(float('-inf'), device=device))
                            starts = torch.argmax(prob_s_rl, dim=1)

                            # starts 内のid がNullを示す255の場合エラーが起きる → starts自体は変更せず，一時的に0をトークンエンベディングとする.(勾配計算は無効かつ，rewardでは無視されるので問題ない)
                            start_tokens = token_embs[torch.arange(batch_size), torch.where(starts==255, torch.tensor(0, device=device), starts)].unsqueeze(1) # 
                            start_tokens_expanded = start_tokens.expand(-1, seq_len-1, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                            
                            # end
                            prob_e_rl = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                            for j, srl_labels in enumerate(batch_srl_labels):
                                filter_e = mk_filter_e(srl_labels, MAX_TOKEN-1, lab2id, starts[j])
                                prob_e_rl[j] = torch.where(filter_e.unsqueeze(0), prob_e_rl[j], torch.tensor(float('-inf'), device=device))
                            ends = torch.argmax(prob_e_rl, dim=1)
                            end_tokens = token_embs[torch.arange(batch_size), ends].unsqueeze(1) # [batch, 1, hidden]
                            
                            # srl
                            # starts 内のid がNullを示す255の場合エラーが起きる → starts自体は変更せず，一時的に0をトークンエンベディングとする.(勾配計算は無効かつ，rewardでは無視されるので問題ない)
                            start_end_vecs = torch.concatenate([start_tokens, srl_embs[torch.arange(batch_size), torch.where(starts==255, torch.tensor(0, device=device), starts)].unsqueeze(1),
                                                                end_tokens, srl_embs[torch.arange(batch_size), ends].unsqueeze(1)], dim=2)
                            prob_srl_rl = srl_decoder(start_end_vecs)
                            srls = torch.argmax(prob_srl_rl, dim=1)
                        
                        """ Making data """
                        batch_target_s, batch_target_e, batch_target_srl = [], [], []
                        targets = []
                        for j in range(batch_size):
                            s, e, srl = target_dicts[j][f'{next_trans[j].item()}'].pop()
                            batch_target_s.append(s), batch_target_e.append(e), batch_target_srl.append(srl)
                            targets.append([s,e,srl])

                        preds = list(zip(starts, ends, srls))
                        g = policy.cal_rewards(preds, targets, MAX_TOKEN)
                        G += torch.tensor(g, device=device)
                        batch_target_s, batch_target_e = torch.tensor(batch_target_s).to(device), torch.tensor(batch_target_e).to(device)
                        batch_target_srl = torch.tensor(batch_target_srl).to(device)
                
                        """ Train """
                        # 開始
                        prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        start_tokens = token_embs[torch.arange(batch_size), batch_target_s] # [batch, hidden]
                        start_tokens_expanded = start_tokens.unsqueeze(1).expand(-1, seq_len-1, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                        
                        # 終了 
                        prob_e = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                        
                        # srl
                        start_end_vecs = torch.concatenate(
                            [token_embs[torch.arange(batch_size), batch_target_s].unsqueeze(1), srl_embs[torch.arange(batch_size), batch_target_s].unsqueeze(1),
                             token_embs[torch.arange(batch_size), batch_target_e].unsqueeze(1), srl_embs[torch.arange(batch_size), batch_target_e].unsqueeze(1)], dim=2)
                        prob_srl = srl_decoder(start_end_vecs)
                        
                        # loss & backprop (SRL)
                        batch_loss_s = loss_function(prob_s, batch_target_s)
                        batch_loss_e = loss_function(prob_e, batch_target_e)
                        batch_loss_srl = loss_function(prob_srl, batch_target_srl)
                        batch_loss_multi = (batch_loss_s + batch_loss_e + batch_loss_srl) / ITERS_TO_ACCUMULATE
                        batch_loss_multi.backward(retain_graph=True)
                        batch_loss_sum += batch_loss_multi.item() 

                        # loss (RL)
                        for g, b, probs, trans in zip(G, G/(i+1), probs_rl, next_trans):
                            batch_policy_loss += 1/batch_size * -(torch.log(probs[trans] + 1e-9)) * (g - b) / ITERS_TO_ACCUMULATE
                    
                        # ラベル更新
                        for j, (s, e, srl) in enumerate(targets):
                            batch_srl_labels[j][s:e+1] = torch.tensor(srl).expand(e-s+1)    # end-start >= 0 は保証されている．
                            
                    # 反復試行の最終であれば
                    else:
                        srl_embs = srl_encoder(batch_srl_labels, batch_attention_masks[:,1:])       # [batch, token, hidden]
                        
                        """ RL """
                        # get next transition
                        probs_rl = policy((batch_arg_indications>=1).int().to(device), F.one_hot(prev_trans, num_classes=len(id2lab)-3).to(device),
                                          torch.concatenate([token_embs.detach(), srl_embs.detach()], dim=2))    # SRLモデルに影響を与えないようにdetach
                        next_trans = torch.tensor([-1]*batch_size)
                        for j, arg_indications in enumerate(batch_arg_indications):
                            filter_srl = mk_filter_srl(arg_indications).to(device)
                            next_trans[j] = torch.argmax(torch.where(filter_srl.unsqueeze(0), probs_rl[j], torch.tensor(float('-inf'), device=device)), dim=1)
                        prev_trans = next_trans

                        with torch.no_grad():   # 計算グラフを保持しつつ，以下の箇所では記録を行わない
                            # start
                            prob_s_rl = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                            for j, srl_labels in enumerate(batch_srl_labels):
                                filter_s = mk_filter_s(srl_labels, MAX_TOKEN, lab2id)
                                prob_s_rl[j] = torch.where(filter_s.unsqueeze(0), prob_s_rl[j], torch.tensor(float('-inf'), device=device))
                            starts = torch.argmax(prob_s_rl, dim=1)
                        
                        """ Makihng data """
                        targets = [[MAX_TOKEN-1]*3]*batch_size
                        preds = list(zip(starts, [-1]*batch_size, [-1]*batch_size))
                        g = policy.cal_rewards(preds, targets, MAX_TOKEN)
                        G += torch.tensor(g, device=device)
                        batch_target_s = torch.tensor([MAX_TOKEN-1]*batch_size).to(device)
                        
                        """ Train """
                        # start
                        prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        
                        # loss & backprop (SRL)
                        batch_loss_s = loss_function(prob_s, batch_target_s)
                        batch_loss_s = batch_loss_s / ITERS_TO_ACCUMULATE
                        batch_loss_s.backward(retain_graph=True)    # srlのiterの累積はしないが，iter_to_accumulateのためにTrueにする．
                        batch_loss_sum += batch_loss_s.item() 
                        epoch_loss += batch_loss_sum

                        # loss & backprop (RL)
                        for g, b, probs, trans in zip(G, G/(i+1), probs_rl, next_trans):
                            batch_policy_loss += 1/batch_size * -(torch.log(probs[trans] + 1e-9)) * (g - b) / ITERS_TO_ACCUMULATE
                            #print()

                        batch_policy_loss.backward(retain_graph=True)
                        policy_loss_sum += batch_policy_loss.item()
                        epoch_policy_loss += policy_loss_sum

                if (i + 1) % ITERS_TO_ACCUMULATE == 0:
                    optimizer.step(), optimizer2.step()
                    sentence_encoder.zero_grad(), srl_encoder.zero_grad()
                    start_decoder.zero_grad(), end_decoder.zero_grad(), srl_decoder.zero_grad()
                    policy.zero_grad()
            print(f"############## epoch {epoch} \t srl_loss {epoch_loss}, rl_loss {epoch_policy_loss} ##############\n")


            """Valid"""
            # Early Stop 判定 + モデル保存判定
            predictions, answers = test(sentence_encoder, srl_encoder, start_decoder, end_decoder, srl_decoder, valid_dataset, lab2id, id2lab)
            ldf, valid_f1 = cal_label_f1(copy.deepcopy(predictions), copy.deepcopy(answers), lab2id)
            print(ldf)
            if prev_f1 <= valid_f1:
                prev_f1 = valid_f1
                patience = 0
                torch.save(sentence_encoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_SentenceEncoder.pth')
                torch.save(srl_encoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_SrlEncoder.pth')
                torch.save(start_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_StartDecoder.pth')
                torch.save(end_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_EndDecoder.pth')
                torch.save(srl_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_SrlDecoder.pth')
                torch.save(policy.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_policy.pth')
                continue
            if patience < 19:
                patience += 1
                print(f'No change in valid acc: patiece {patience}/20\n')
            else:
                print('Early Stop\n')
                break

        print('Train Time = ', (time.time() - start_time) / 60)
    
    
    """Test"""
    sentence_encoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SentenceEncoder.pth'))
    srl_encoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SrlEncoder.pth'))
    start_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_StartDecoder.pth'))
    end_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_EndDecoder.pth'))
    srl_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SrlDecoder.pth'))
    
    # 推論，評価
    predictions, answers = test(sentence_encoder, srl_encoder, start_decoder, end_decoder, srl_decoder, test_dataset, lab2id, id2lab)
    ldf, lf1 = cal_label_f1(copy.deepcopy(predictions), copy.deepcopy(answers), lab2id)
    sdf, sf1 = cal_span_f1(copy.deepcopy(predictions), copy.deepcopy(answers), lab2id, MAX_TOKEN)
    print(ldf, '\n')
    print(sdf)

    # 解析用データ作成
    test_df = pd.concat([group for _, group in groups])    # test_df は group化されている．
    pred_seq = [span2seq(p, int(num_of_tokens)) for p, num_of_tokens in zip(predictions, test_df['num_of_tokens'].to_list())]
    pred_dic_lists, match_count_list, args_count_list = get_pred_dic_lists(copy.deepcopy(predictions), copy.deepcopy(answers), lab2id)
    
    test_df['BIOtag'] = pred_seq
    test_df['pred_arg'] = pred_dic_lists
    test_df['match_count'] = match_count_list
    test_df['args_num'] = args_count_list
    test_df['predict_num'] = [len(dic) for dic in pred_dic_lists]
    test_df = test_df[['sentence', 'sentenceID', 'predicate','args', 'BIOtag', 'pred_arg', 'match_count', 'args_num', 'predict_num']]
    test_df.to_json('data_for_analy.json', orient='records', lines=True, force_ascii=False)