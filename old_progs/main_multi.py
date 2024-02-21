import sys
import time
from tqdm import tqdm
import os


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertConfig
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from itertools import zip_longest

sys.path.append('../')
from network import SentenceEncoder, SRLEncoder, StartDecoder, EndDecoder, SRLDecoder
from prepare_dataset import split2ttv, mk_dataset
from evaluate import cal_label_f1, cal_span_f1, cal_f1_for_decode


MAX_TOKEN = 256     # CLSを含めた次元．データセット内の最大トークンは252
BATCH_SIZE = 32     # 大きい方がよい?
ITERS_TO_ACCUMULATE = 1
MAX_ITER = 7        # データセット内最大 srl 数．
DATAPATH = 'Data/common_data_v2_bert.json'
PRETRAINED_MODEL = "cl-tohoku/bert-base-japanese-v2"
VERSION = 'random'
MODEL_NAME = 'batch32_commonv2'
TRAIN_FLAG = True
MODEL_PATH = f"models/{PRETRAINED_MODEL.replace('/','')}/{VERSION}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


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
    #print(lab2id)

    # 各種データ作成（学習，テスト，検証
    #train_df, test_df, valid_df = split2ttv(df)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train_df, test_valid_df = train_test_split(df, test_size=0.2, random_state=0)
    test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, random_state=0)
    print(f"Data num. Train:{len(train_df)}, Test:{len(test_df)}, Valid:{len(valid_df)}\n")
    
    # テスト時のデータ削減（本実験ではコメントアウト）
    # train_df = train_df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    # valid_df = valid_df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    # test_df = test_df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    
    """ Dataset """
    if TRAIN_FLAG:
        train_dataset = mk_dataset(train_df, BATCH_SIZE, PRETRAINED_MODEL, MAX_TOKEN, lab2id)
        valid_dataset = mk_dataset(valid_df, 1, PRETRAINED_MODEL, MAX_TOKEN, lab2id)
    test_dataset = mk_dataset(test_df, 1, PRETRAINED_MODEL, MAX_TOKEN, lab2id)
    print("Making Dataset Done")

    
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
        num_hidden_layers=1,     # Transformerのレイヤー数
        hidden_size=768*2+256*2, # token_s + srl + token_e + srl
        num_attention_heads=32   # 1つのヘッドが64になるように設定
    )
    sentence_encoder = SentenceEncoder(PRETRAINED_MODEL).to(device)
    srl_encoder = SRLEncoder(srl_config, len(lab2id)).to(device)
    start_decoder = StartDecoder(decoder_config1, MAX_TOKEN).to(device)    # CLSを抜かした255トークンの位置 ＋ null の位置 ＝ 256次元（MAX_TOKEN）
    end_decoder = EndDecoder(decoder_config2, MAX_TOKEN-1).to(device)
    label_decoder = SRLDecoder(decoder_config3, len(lab2id)-3).to(device)  # len(lab2id)-3 なのは，N, V，PAD が 本物のSRLラベルではないから

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
            {'params': label_decoder.parameters(), 'lr': 1e-4}]
        )


        """Train"""
        start = time.time()
        patience = 0
        prev_f1, prev_start_acc, prev_end_acc, prev_srl_acc = 0, 0, 0, 0
        for epoch in range(40):
            epoch_loss = 0
            sentence_encoder.train(), srl_encoder.train(), start_decoder.train(), end_decoder.train(), label_decoder.train()
            #for batch_ids, batch_attention_masks, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl in tqdm(train_dataset):
            for batch_ids, batch_attention_masks, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl in train_dataset:
                #print('iter_batch_srl_labels = ', iter_batch_srl_labels.shape)
                # gpuへ転送
                batch_ids, batch_attention_masks = batch_ids.to(device), batch_attention_masks.to(device)
                iter_batch_srl_labels = iter_batch_srl_labels.to(device)
                iter_target_s, iter_target_e, iter_target_srl = iter_target_s.to(device), iter_target_e.to(device), iter_target_srl.to(device)

                token_embs = sentence_encoder(batch_ids, batch_attention_masks)                     # batch, token, hidden
                batch_loss_sum = 0
                for i, (batch_srl_labels, batch_target_s, batch_target_e, batch_target_srl) in enumerate(
                        zip(iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl)): # e, srl はNoneで埋め合わされる
                    # 反復試行の最終でないなら
                    #print(token_embs.shape)
                    iter_size, batch_size, token_len = iter_batch_srl_labels.shape
                    if i != iter_size-1:
                        srl_embs = srl_encoder(batch_srl_labels, batch_attention_masks[:,1:])       # [batch, token, hidden]
                        
                        # 開始
                        prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        start_tokens = token_embs[torch.arange(batch_size), batch_target_s] # [batch, hidden]
                        start_tokens_expanded = start_tokens.unsqueeze(1).expand(-1, token_len, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                        #print('start_tokens', start_tokens.shape)
                        
                        # 終了 
                        prob_e = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                        
                        # srl
                        start_end_vecs = torch.concatenate(
                            [token_embs[torch.arange(batch_size), batch_target_s].unsqueeze(1), srl_embs[torch.arange(batch_size), batch_target_s].unsqueeze(1),
                             token_embs[torch.arange(batch_size), batch_target_e].unsqueeze(1), srl_embs[torch.arange(batch_size), batch_target_e].unsqueeze(1)], dim=2)
                        prob_srl = label_decoder(start_end_vecs)
                        
                        # backprop
                        batch_loss_s = loss_function(prob_s, batch_target_s)
                        batch_loss_e = loss_function(prob_e, batch_target_e)
                        batch_loss_srl = loss_function(prob_srl, batch_target_srl)
                        batch_loss_multi = (batch_loss_s + batch_loss_e + batch_loss_srl) / 3
                        batch_loss_multi.backward(retain_graph=True)
                        batch_loss_sum += batch_loss_s.item() + batch_loss_e.item() + batch_loss_srl.item()
                    else:
                        #print('Last Iter')
                        srl_embs = srl_encoder(batch_srl_labels, batch_attention_masks[:,1:])       # [batch, token, hidden]
                        prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        batch_loss_s = loss_function(prob_s, batch_target_s)
                        batch_loss_s.backward(retain_graph=True)
                        
                if (i + 1) % ITERS_TO_ACCUMULATE == 0:
                    optimizer.step()
                    sentence_encoder.zero_grad()
                    srl_encoder.zero_grad()
                    start_decoder.zero_grad()
                    end_decoder.zero_grad()
                    label_decoder.zero_grad()
                    epoch_loss += batch_loss_sum / ITERS_TO_ACCUMULATE
            print(f"############## epoch {epoch} \t loss {epoch_loss} ##############\n")


            """Valid"""
            valid_f1, start_acc, end_acc, srl_acc = 0, 0, 0, 0
            predictions, answers = [], []
            pred_start_list, pred_end_list, pred_srl_list = [], [], []
            sentence_encoder.eval(), srl_encoder.eval(), start_decoder.eval(), end_decoder.eval(), label_decoder.eval()
            with torch.no_grad():
                #for batch_ids, batch_attention_masks, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl in tqdm(valid_dataset):
                for batch_ids, batch_attention_masks, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl in valid_dataset:
                    batch_ids, batch_attention_masks = batch_ids.to(device), batch_attention_masks.to(device)
                    iter_target_s, iter_target_e, iter_target_srl = iter_target_s.to(device), iter_target_e.to(device), iter_target_srl.to(device)

                    # 正解データ作成
                    answer, prediction, pred_start, pred_end, pred_srl = [], [], [], [], []
                    for batch_target_s, batch_target_e, batch_target_srl in zip(iter_target_s, iter_target_e, iter_target_srl):
                        for target_s, target_e, target_srl in zip(batch_target_s, batch_target_e, batch_target_srl):
                            answer.append((target_s, target_e, target_srl))
                    answers.append(answer)

                    # 推測
                    token_embs = sentence_encoder(batch_ids, batch_attention_masks)         # batch, token, hidden
                    srl_labels = [lab2id['N']] * (len(batch_attention_masks[0]) - 1)        # srl_label の初期値
                    srl_labels = torch.tensor([srl_labels]).to(device)
                    batch_size, token_len, hidden = token_embs.shape
                    #print(srl_labels.shape)
                    # start に Null を予測するまではループ
                    for iter in range(MAX_ITER):
                        srl_embs = srl_encoder(srl_labels, batch_attention_masks[:,1:])     # iなのは1データずつだから
                        
                        # 開始位置
                        prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                        start = torch.argmax(prob_s[:, :token_len], dim=1)[0]  # token_len でフィルターを掛ける．
                        #print('start = ', start)
                        #print('start and pad ligits = ', [prob_s[0, start], prob_s[0, -1]])
                        start = start if torch.argmax(torch.tensor([prob_s[0, start], prob_s[0, -1]])) == 0 else MAX_TOKEN-1    # Null とどちらが大きいか
                        if start == MAX_TOKEN-1:              # 最終次元はNullを示す．-1 なのはargmaxが理由
                            break
                        start_tokens = token_embs[torch.arange(batch_size), [start]].unsqueeze(1) # [batch, 1, hidden]
                        start_tokens_expanded = start_tokens.expand(-1, token_len, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                        
                        # 終了位置
                        #print(token_embs.shape, srl_embs.shape, start_tokens_expanded.shape)
                        prob_e = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                        end = start + torch.argmax(prob_e[:, start:token_len], dim=1)[0]    # start と token_len でフィルターを掛ける．endは最大で254. startをプラスしてフィルターによる差を消す．
                        end_tokens = token_embs[torch.arange(batch_size), [end]].unsqueeze(1) # [batch, 1, hidden]
                        
                        # 意味役割と次回のiter用にsrl_labelsを更新
                        start_end_vecs = torch.concatenate([start_tokens, srl_embs[torch.arange(batch_size), [start]].unsqueeze(1),
                                                            end_tokens, srl_embs[torch.arange(batch_size), [end]].unsqueeze(1)], dim=2)
                        prob_srl = label_decoder(start_end_vecs)
                        
                        srl = torch.argmax(prob_srl, dim=1)
                        srl_labels[0][start:end+1] = srl.expand(end-start+1)    # end-start >= 0 は保証されている．
                        #print(start, end, srl)
                        
                        # 正解と予測の作成
                        prediction.append((start, end, srl))
                        pred_start.append(start), pred_end.append(end), pred_srl.append(srl)
                    pred_start_list.append(pred_start), pred_end_list.append(pred_end), pred_srl_list.append(pred_srl)
                    predictions.append(prediction)

            ## モデル保存判定
            #start_acc = cal_f1_for_decode(valid_dataset[3].reshape(-1), prediction_start)
            #if prev_start_acc <= start_acc:
            #    prev_start_acc = start_acc
            #    torch.save(start_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_StartDecoder.pth')
            #
            #end_acc = cal_f1_for_decode(valid_dataset[4].reshape(-1), prediction_end)
            #if prev_end_acc <= end_acc:
            #    prev_end_acc = end_acc
            #    torch.save(end_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_EndDecoder.pth')
            #
            #srl_acc = cal_f1_for_decode(valid_dataset[5].reshape(-1), prediction_srl)
            #if prev_srl_acc <= srl_acc:
            #    prev_srl_acc = srl_acc
            #    torch.save(label_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_LabelDecoder.pth')

            # Early Stop 判定 + モデル保存判定
            ldf, valid_f1 = cal_label_f1(predictions, answers, lab2id)
            print(ldf)
            if prev_f1 <= valid_f1:
                prev_f1 = valid_f1
                patience = 0
                torch.save(sentence_encoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_SentenceEncoder.pth')
                torch.save(srl_encoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_SrlEncoder.pth')
                
                # 以下は実験的
                torch.save(start_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_StartDecoder.pth')
                torch.save(end_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_EndDecoder.pth')
                torch.save(label_decoder.state_dict(), f'{MODEL_PATH}/{MODEL_NAME}_LabelDecoder.pth')
                continue
            if patience < 9:
                patience += 1
                print(f'No change in valid acc: patiece {patience}/10\n')
            else:
                print('Early Stop\n')
                break

        """Train end"""
        print('Train Time = ', (time.time()-start)/60)
    
    
    """Test"""
    #classifier = BertClassifier(PRETRAINED_MODEL, OUTPUT_LAYER_DIM).to(device)
    sentence_encoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SentenceEncoder.pth'))
    srl_encoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_SrlEncoder.pth'))
    start_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_StartDecoder.pth'))
    end_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_EndDecoder.pth'))
    label_decoder.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_LabelDecoder.pth'))
    
    predictions, answers = [], []
    pred_start_list, pred_end_list, pred_srl_list = [], [], []
    sentence_encoder.eval(), srl_encoder.eval(), start_decoder.eval(), end_decoder.eval(), label_decoder.eval()
    with torch.no_grad():
        #for batch_ids, batch_attention_masks, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl in tqdm(test_dataset):
        for batch_ids, batch_attention_masks, iter_batch_srl_labels, iter_target_s, iter_target_e, iter_target_srl in test_dataset:
            batch_ids, batch_attention_masks = batch_ids.to(device), batch_attention_masks.to(device)
            iter_target_s, iter_target_e, iter_target_srl = iter_target_s.to(device), iter_target_e.to(device), iter_target_srl.to(device)

            # 正解データ作成
            answer, prediction, pred_start, pred_end, pred_srl = [], [], [], [], []
            for batch_target_s, batch_target_e, batch_target_srl in zip(iter_target_s, iter_target_e, iter_target_srl):
                for target_s, target_e, target_srl in zip(batch_target_s, batch_target_e, batch_target_srl):
                    answer.append((target_s, target_e, target_srl))
            answers.append(answer)

            # 推測
            token_embs = sentence_encoder(batch_ids, batch_attention_masks)         # batch, token, hidden
            srl_labels = [lab2id['N']] * (len(batch_attention_masks[0]) - 1)        # srl_label の初期値
            srl_labels = torch.tensor([srl_labels]).to(device)
            batch_size, token_len, hidden = token_embs.shape
            #print(srl_labels.shape)
            # start に Null を予測するまではループ
            for iter in range(MAX_ITER):
                srl_embs = srl_encoder(srl_labels, batch_attention_masks[:,1:])     # iなのは1データずつだから
                
                # 開始位置
                prob_s = start_decoder(torch.concatenate([token_embs, srl_embs], dim=2), batch_attention_masks[:,1:])
                start = torch.argmax(prob_s[:, :token_len], dim=1)[0]  # token_len でフィルターを掛ける．
                start = start if torch.argmax(torch.tensor([prob_s[0, start], prob_s[0, -1]])) == 0 else MAX_TOKEN-1    # Null とどちらが大きいか
                if start == MAX_TOKEN-1:              # 最終次元はNullを示す．-1 なのはargmaxが理由
                    break
                start_tokens = token_embs[torch.arange(batch_size), [start]].unsqueeze(1) # [batch, 1, hidden]
                start_tokens_expanded = start_tokens.expand(-1, token_len, -1) # [:,target,:]と[[0,1,...],target]は，別物であることに注意
                
                # 終了位置
                #print(token_embs.shape, srl_embs.shape, start_tokens_expanded.shape)
                prob_e = end_decoder(torch.concatenate([token_embs, srl_embs, start_tokens_expanded], dim=2), batch_attention_masks[:,1:])
                end = start + torch.argmax(prob_e[:, start:token_len], dim=1)[0]    # start と token_len でフィルターを掛ける．endは最大で254. startをプラスしてフィルターによる差を消す．
                end_tokens = token_embs[torch.arange(batch_size), [end]].unsqueeze(1) # [batch, 1, hidden]
                
                # 意味役割と次回のiter用にsrl_labelsを更新
                start_end_vecs = torch.concatenate([start_tokens, srl_embs[torch.arange(batch_size), [start]].unsqueeze(1),
                                                    end_tokens, srl_embs[torch.arange(batch_size), [end]].unsqueeze(1)], dim=2)
                prob_srl = label_decoder(start_end_vecs)
                
                srl = torch.argmax(prob_srl, dim=1)
                srl_labels[0][start:end+1] = srl.expand(end-start+1)    # end-start >= 0 は保証されている．
                
                # 正解と予測の作成
                prediction.append((start, end, srl))
                pred_start.append(start), pred_end.append(end), pred_srl.append(srl)
            pred_start_list.append(pred_start), pred_end_list.append(pred_end), pred_srl_list.append(pred_srl)
            predictions.append(prediction)

    ldf, lf1 = cal_label_f1(predictions, answers, lab2id)
    sdf, sf1 = cal_span_f1(predictions, answers, lab2id, MAX_TOKEN)
    print(ldf)
    print(sdf)