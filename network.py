import torch
from torch import nn
from transformers import BertModel, BertConfig


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config, output_attentions=True, output_hidden_states=True)

    def _get_token_vecs(self, vec):
        return vec[:, 1:, :]    # batch, sent_token, 768

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask = attention_mask)
        hidden_states = output['hidden_states']

        # 文章の各トークンベクトルを取得する
        token_vecs = self._get_token_vecs(hidden_states[-1])

        return token_vecs
    
    
class SRLEncoder(nn.Module):
    def __init__(self, config, num_labels):
        super(SRLEncoder, self).__init__()
        self.bert = BertModel(config)
        self.label_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=config.hidden_size)  # ラベルの埋め込み定義

    def forward(self, srl_labels, attention_mask):
        custom_embeddings = self.label_embedding(srl_labels)  # [batch, seq_len, hidden]
        #print(srl_labels.shape, custom_embeddings.shape, attention_mask.shape)
        output = self.bert(inputs_embeds=custom_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output['hidden_states']

        # 文章の各トークンベクトルを取得する
        srl_vecs = hidden_states[-1]    # batch, token, hidden
        return srl_vecs


class SRLEncoder2(nn.Module):
    def __init__(self, config, num_labels):
        super(SRLEncoder2, self).__init__()
        self.bert = BertModel(config)
        self.label_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=int(config.hidden_size/2))  # ラベルの埋め込み定義

    def forward(self, batch_srl_labels, batch_srl_prev_labels, attention_mask):
        srl_embeddings = self.label_embedding(batch_srl_labels)  # [batch, seq_len, hidden]
        srl_prev_embeddings = []
        for srl_prev_labels in batch_srl_prev_labels:
            # 埋め込みを取得し、述語の次元で平均を計算
            embeddings = self.label_embedding(srl_prev_labels).mean(dim=0)
            srl_prev_embeddings.append(embeddings.unsqueeze(0))  # バッチ次元の形状を保持
        srl_prev_embeddings = torch.cat(srl_prev_embeddings, dim=0)
        
        output = self.bert(inputs_embeds=torch.cat([srl_embeddings, srl_prev_embeddings], dim=2), attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = output['hidden_states']

        # 文章の各トークンベクトルを取得する
        srl_vecs = hidden_states[-1]    # batch, token, hidden
        return srl_vecs


class StartDecoder(nn.Module):
    def __init__(self, config, output_dim):
        super(StartDecoder, self).__init__()
        self.bert = BertModel(config)
        self.linear_in_dim = config.hidden_size*(output_dim-1)
        self.linear = nn.Linear(self.linear_in_dim, output_dim)    # input, output
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    def _get_token_vecs(self, vec):
        return vec[:, 1:, :]    # batch, sent_token, hidden
    
    def forward(self, custom_embeddings, attention_mask=None):
        outputs = self.bert(inputs_embeds=custom_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
        token_vecs = self._get_token_vecs(hidden_states[-1])
        
        # Calculate the total size needed for padding
        batch_size, seq_len, hidden_size = token_vecs.size()
        
        # Reshape and pad if necessary
        token_vecs_flat = token_vecs.reshape(batch_size, -1)
        pad_size = self.linear_in_dim - token_vecs_flat.size(1)
        
        if pad_size > 0:
            pad = torch.zeros(batch_size, pad_size, device=token_vecs.device)
            token_vecs_flat = torch.cat([token_vecs_flat, pad], dim=1)
            
        logits = self.linear(token_vecs_flat)
        results = self.logSoftmax(logits)  # outs[b_idx, max, label]

        return results


class EndDecoder(nn.Module):
    def __init__(self, config, output_dim):
        super(EndDecoder, self).__init__()
        self.bert = BertModel(config)
        self.linear_in_dim = config.hidden_size*output_dim
        self.linear = nn.Linear(self.linear_in_dim, output_dim)    # input, output
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    def _get_token_vecs(self, vec):
        return vec[:, 1:, :]    # batch, sent_token, hidden
    
    def forward(self, custom_embeddings, attention_mask=None):
        outputs = self.bert(inputs_embeds=custom_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
        token_vecs = self._get_token_vecs(hidden_states[-1])
        
        # Calculate the total size needed for padding
        batch_size, seq_len, hidden_size = token_vecs.size()
        
        # Reshape and pad if necessary
        token_vecs_flat = token_vecs.reshape(batch_size, -1)
        pad_size = self.linear_in_dim - token_vecs_flat.size(1)
        
        if pad_size > 0:
            pad = torch.zeros(batch_size, pad_size, device=token_vecs.device)
            token_vecs_flat = torch.cat([token_vecs_flat, pad], dim=1)
            
        logits = self.linear(token_vecs_flat)
        results = self.logSoftmax(logits)  # outs[b_idx, max, label]

        return results


class SRLDecoder(nn.Module):
    def __init__(self, config, output_dim):
        super(SRLDecoder, self).__init__()
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, output_dim)    # input, output
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    def forward(self, custom_embeddings, attention_mask=None):
        outputs = self.bert(inputs_embeds=custom_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
        token_vec = hidden_states[-1][:,0,:]
        
        logits = self.linear(token_vec.reshape(len(token_vec), -1))
        results = self.logSoftmax(logits)  # outs[b_idx, max, label]

        return results


class GumbelSoftmaxPolicy(nn.Module):
    def __init__(self, n_actions, hidden_size, seq_len):
        super(GumbelSoftmaxPolicy, self).__init__()
        self.tau = torch.tensor(0.1)                              # 温度．0.1, 0.5, 1.0 でテスト
        self.linear_in_dim = hidden_size*seq_len
        self.linear = nn.Linear(self.linear_in_dim, n_actions-1)    # input, output, srlラベルならn_action-1
        self.linear2 = nn.Linear((n_actions-1)*3, n_actions)            # input, output
        
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear2.bias, 0)
        
    def forward(self, states, prev_arg_indication, embeddings):
        """
        Args:
            states (): 
            custom_embeddings (): 意味役割ラベルを埋め込んだトークエンベディング
            
        Returns:
            logits ():
            next_trans ():
        """
        # Calculate the total size needed for padding
        batch_size, _, _ = embeddings.size()
        
        # Reshape and pad if necessary
        token_vecs_flat = embeddings.reshape(batch_size, -1)
        pad_size = self.linear_in_dim - token_vecs_flat.size(1)
        
        if pad_size > 0:
            pad = torch.zeros(batch_size, pad_size, device=embeddings.device)
            token_vecs_flat = torch.cat([token_vecs_flat, pad], dim=1)

        output = self.linear(token_vecs_flat)
        logits = self.linear2(torch.cat([output, states, prev_arg_indication], dim=1))
        gumbel_noise = torch.distributions.gumbel.Gumbel(loc=0.0, scale=1.0).sample(logits.shape).to(embeddings.device)
        probs = nn.functional.softmax((logits + gumbel_noise) / self.tau, dim=0)
        #next_trans = torch.argmax(probs, dim=1)
            
        return probs

    def predict(self, states, prev_arg_indication, embeddings):
        """
        Args:
            states (): 
            custom_embeddings (): 意味役割ラベルを埋め込んだトークエンベディング
            
        Returns:
            next_trans ():
        """
        # Calculate the total size needed for padding
        batch_size, _, _ = embeddings.size()
        
        # Reshape and pad if necessary
        token_vecs_flat = embeddings.reshape(batch_size, -1)
        pad_size = self.linear_in_dim - token_vecs_flat.size(1)
        
        if pad_size > 0:
            pad = torch.zeros(batch_size, pad_size, device=embeddings.device)
            token_vecs_flat = torch.cat([token_vecs_flat, pad], dim=1)

        output = self.linear(token_vecs_flat)
        logits = self.linear2(torch.cat([output, states, prev_arg_indication], dim=1))
        probs = nn.functional.softmax(logits, dim=0)
        next_trans = torch.argmax(probs, dim=1)
        
        return next_trans
    
    def cal_rewards(self, preds, targets, max_token):
        rewards = []
        for i, (pred, targ) in enumerate(zip(preds, targets)):
            reward = 0
            p_start, p_end, p_srl = pred
            t_start, t_end, t_srl = targ
            # start がNullでないなら
            if t_start != max_token-1:
                if p_start == t_start:
                    reward += 1
                else:
                    reward -= 1
                    rewards.append(reward)
                    continue
                
                if p_end == t_end:
                    reward += 1
                else:
                    reward -= 1
                    rewards.append(reward)
                    continue
                
                if p_srl == t_srl:
                    reward += 1
                else:
                    reward -= 1
                rewards.append(reward)
                    
            # Nullが推測できれば+1．ペナルティなし
            else:
                if p_start == t_start:
                    reward += 1
                rewards.append(reward)
        assert len(targets) == len(rewards)
            
        return rewards


if __name__ == '__main__':
    print('Test')