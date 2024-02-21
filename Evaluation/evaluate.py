import numpy as np
import pandas as pd


def cal_label_f1(xs, ys, lab2id):
    """意味役割に関する評価を計算する
    args:
        xs (list): 予測データ．各データにおける意味役割のタプルを格納したリスト
        ys (list): 正解データ．各データにおける意味役割のタプルを格納したリスト
        lab2id (dict): 意味役割と「V」,「PAD」,「N」を含むラベル辞書
    
    Returns:
        df: 評価のdf
        f1: f1値
    """
    num_srl_labels = len(lab2id) - 3
    array = np.zeros(num_srl_labels*5).reshape(num_srl_labels, 5)
    for span_label_pre_lists, span_label_ans_lists in zip(xs, ys):
        for _, _, y_lab in span_label_ans_lists:
            y_lab_id = lab2id[y_lab]
            array[y_lab_id][4] += 1   # support + 1

        # 正解の項が（start, end, srl）が存在するかどうか
        for pre in span_label_pre_lists:
            p_lab_id = lab2id[pre[2]]
            if pre in span_label_ans_lists:
                array[p_lab_id][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
            else:
                array[p_lab_id][1] += 1 # wrong_pred + 1
        
        for ans in span_label_ans_lists:
            y_lab_id = lab2id[ans[2]]
            array[y_lab_id][3] += 1 # wrong_missed + 1

    array = array.T
    array[2] = array[0] + array[1]
    array = array.T
    
    # 各ラベルの評価
    df = pd.DataFrame(array, columns=['correct_num', 'wrong_num', 'predict_num', 'missed_num', 'support'], index=list(lab2id.keys())[:-3])
    df['precision'] = df['correct_num'] / df['predict_num']
    df['recall'] = df['correct_num'] / df['support']
    df['f1'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    
    # 以下全体
    df.loc['sum'] = df.sum()
    if df.loc['sum', 'predict_num'] == 0:
        df.loc['sum', 'predict_num'] = 1
    
    df.loc['precision','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'predict_num']
    df.loc['recall','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'support']
    df.loc['f1','correct_num'] = 2*df.loc['precision','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision','correct_num'] + df.loc['recall','correct_num'])
    df.loc['f1_macro','correct_num'] = df['f1'].sum() / num_srl_labels
    
    df = df.round(4)
    f1 = df.loc['f1','correct_num'] if not np.isnan(df.loc['f1','correct_num']) else 0
    return df, f1


def cal_span_f1(xs, ys, lab2id, max_len):
    """スパン幅に関する評価を計算する
    args:
        xs (list): 予測データ．各データにおける意味役割のタプルを格納したリスト
        ys (list): 正解データ．各データにおける意味役割のタプルを格納したリスト
        max_len (int): スパン幅の最大値．実際はシーケンス長が渡されることになる
    
    Returns:
        df: 評価のdf
        f1: f1値
    """
    array = np.zeros(max_len*5).reshape(max_len, 5)
    for span_label_pre_lists, span_label_ans_lists in zip(xs, ys):
        for y_start, y_end, _ in span_label_ans_lists:
            array[y_end-y_start][4] += 1   # support + 1

        for pre in span_label_pre_lists:
            if pre in span_label_ans_lists:
                array[pre[1]-pre[0]][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
            else:
                array[pre[1]-pre[0]][1] += 1 # wrong_pred + 1
        
        for ans in span_label_ans_lists:
            array[ans[1]-ans[0]][3] += 1 # wrong_missed + 1

    array = array.T
    array[2] = array[0] + array[1]
    array = array.T
    
    # 各ラベルの評価
    df = pd.DataFrame(array, columns=['correct_num', 'wrong_num', 'predict_num', 'missed_num', 'support'], index=np.arange(1,max_len+1))
    df['precision'] = df['correct_num'] / df['predict_num']
    df['recall'] = df['correct_num'] / df['support']
    df['f1'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    
    # 以下全体
    df.loc['sum'] = df.sum()
    if df.loc['sum', 'predict_num'] == 0:
        df.loc['sum', 'predict_num'] = 1
    df.loc['precision','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'predict_num']
    df.loc['recall','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'support']
    df.loc['f1','correct_num'] = 2*df.loc['precision','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision','correct_num'] + df.loc['recall','correct_num'])
    #df.loc['f1_macro','correct_num'] = df['f1'].sum() / len(lab2id)
    df = df.round(4)
    f1 = df.loc['f1','correct_num'] if not np.isnan(df.loc['f1','correct_num']) else 0
    return df, f1


def seq_fusion_matrix(xs, ys, lab2id):
    """混合行列の計算
    spanが正解していてラベルを間違えたものの混合行列を計算する．
    
    Args:
        xs(各文章毎のラベル結果（start, end, label）のリスト) : 予測．不必要なラベルも含まれる
        ys(各文章毎のラベル結果（start, end, label）のリスト) : 正解．
    
    Returns:
        DataFrame : 混合行列
    """

    array = np.zeros(len(lab2id)*(len(lab2id))).reshape(len(lab2id), len(lab2id))
    for span_label_pre_lists, span_label_ans_lists in zip(xs, ys):
        for pre in span_label_pre_lists:
            for ans in span_label_ans_lists:
                if pre[0] == ans[0] and pre[1] == ans[1]:   # spanが正解していてラベルを間違えたものの混合行列
                    if  pre[2] != ans[2]:
                        array[lab2id[ans[2]]][lab2id[pre[2]]] += 1

    return pd.DataFrame(array, columns= list(lab2id.keys()), index= list(lab2id.keys()))


def get_pred_dic_lists(xs, ys, lab2id):
    pred_dic_lists = []
    match_count_list = []
    args_count_list = []
    assert len(xs) == len(ys)
    array = np.zeros(len(lab2id)*5).reshape(len(lab2id), 5)
    for span_label_pre_lists, span_label_ans_lists in zip(xs, ys):
        pred_dic = []
        match_count = 0
        # データ成型, 予想spanのラベルにフラグメントと述語は含まれない．
        args_count_list.append(len(span_label_ans_lists))
        for _, _, y_lab in span_label_ans_lists:
            y_lab_id = lab2id[y_lab]
            array[y_lab_id][4] += 1   # support + 1

        for pre in span_label_pre_lists:
            p_lab_id = lab2id[pre[2]]
            if pre in span_label_ans_lists:
                array[p_lab_id][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
                pred_dic.append({"start":pre[0], "end":pre[1], "role":pre[2], "true_false":True})
                match_count += 1
            else:
                array[p_lab_id][1] += 1 # wrong_pred + 1
                pred_dic.append({"start":pre[0], "end":pre[1], "role":pre[2], "true_false":False})

        for ans in span_label_ans_lists:
            y_lab_id = lab2id[ans[2]]
            array[y_lab_id][3] += 1 # wrong_missed + 1
        pred_dic_lists.append(pred_dic)
        match_count_list.append(match_count)

    return pred_dic_lists, match_count_list, args_count_list


def span2seq(spans, token_len):
    seq = ['O'] * token_len
    for i, j, arg in spans:
        if arg != 'O':
            seq[i] = f'B-{arg}'
            for dis in range(j-i):
                seq[i+dis+1] = f'I-{arg}'

    return seq    