import json

def cal_f1(match_count,pred_count,true_count):
        if pred_count != 0:
            precision = match_count*100 / pred_count
        else:
            precision = 0

        if true_count != 0: # 実際 true_count　== 0 はゼロ件
            recall = match_count*100 / true_count
        else:
            recall = 0

        if recall != 0 and precision != 0:
            f1 = 2*precision*recall / (precision+recall)
        else:
            f1 = 0
        return f1

def main():
    try:
        with open("data_for_analy.json") as f: 
            test_json_list = json.load(f)
    except:
        data = []
        with open('data_for_analy.json', 'r') as f:
            for line in f:
                data.append(json.loads(line))
        with open(f"data_for_analy.json", mode="w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open("data_for_analy.json") as f: 
            test_json_list = json.load(f)
    
    # match_count, predict_count
    all_oracle_f1 = {"fix_labels":0,"move_arg":0,"merge_span":0,"split_span":0,"fix_border":0,"remove_arg":0,"add_arg":0}
    sentence_ids = {"fix_labels":[],"move_arg":[],"merge_span":[],"split_span":[],"fix_border":[],"remove_arg":[],"add_arg":[],"other_pred":[],"other_ans":[]}

    
    core_dic = {"Arg0":0,"Arg1":1,"Arg2":2,"Arg3":3,"Arg4":4,"Arg5":5} # listでも良い 速度上げるためにdict


    for json_ in test_json_list:
        true_dic = {}
        pred_dic = {}
        oracle = {"fix_labels":0,"move_arg":0,"merge_span":0,"split_span":0,"fix_border":0,"remove_arg":0,"add_arg":0}
        # answer_count += json_["args_num"]
        # predict_count += json_["predict_num"]
        # match_count += json_["match_count"]

        true_sentence = [0]*len(json_["sentence"].split())
        for i,arg_ in enumerate(json_["args"],5):
            if arg_["word_start"] == -1:
                continue
            true_dic[arg_["word_start"]] = [arg_["argrole"],arg_["word_end"]]
            true_sentence[arg_["word_start"]:arg_["word_end"]+1] = [1]*(arg_["word_end"]+1-arg_["word_start"])

        # for tkey,tval in true_dic.items():
        #     true_sentence[tkey:tval[1]+1] = [1]*len(tval[1]+1-tkey)
        
        pred_sentence = [0]*len(json_["sentence"].split())
        for i,arg_ in enumerate(json_["pred_arg"],1):
            pred_sentence[arg_["start"]:arg_["end"]+1] = [1]*(arg_["end"]+1-arg_["start"])
            if arg_["true_false"]:
                true_dic.pop(arg_["start"])
                continue
            pred_dic[arg_["start"]] = [arg_["role"],arg_["end"]]
        
        # pred_sentence = [0]*len(json_["sentence"].split())
        # for pkey,pval in pred_dic.items():
        #     pred_sentence[pkey:pval[1]+1] = [1]*len(pval[1]+1-pkey)
        
        # fix labels
        pred_keys = list(pred_dic.keys())
        for pkey in pred_keys:
            pval = pred_dic[pkey]
            if pkey in true_dic: # 開始位置が一致
                tval = true_dic[pkey]
                if pval[1] == tval[1]: # 終了位置が一致
                    oracle["fix_labels"] += 1
                    pred_dic.pop(pkey)
                    true_dic.pop(pkey)
                    sentence_ids["fix_labels"].append(json_["sentenceID"]+"_"+str(pkey))
                    # 正解　+1
        
        # move arg
        pred_keys = list(pred_dic.keys())
        for pkey in pred_keys:
            pval = pred_dic[pkey]
            if pval[0] in core_dic: # 予測したラベルがcore
                for tkey,tval in true_dic.items():
                    if pval[0] == tval[0] and (sum(pred_sentence[tkey:tval[1]+1]) == 0):# or pkey <= tval[1]): # 正解していない同じコアラベルがあり、 その部分に別の予測がされていない(自分の予測ならok)
                        oracle["move_arg"] += 1
                        pred_dic.pop(pkey)
                        true_dic.pop(tkey)
                        sentence_ids["move_arg"].append(json_["sentenceID"]+"_"+str(pkey))
                        # 正解　+1
                        break
        
        # merge span
        true_keys = list(true_dic.keys())
        for tkey in true_keys:
            tval = true_dic[tkey]
            if tkey in pred_dic: # 正解の開始位置と同じ開始位置の予測が有る
                frontval = pred_dic[tkey]
                pred_keys = list(pred_dic.keys())
                for backkey in pred_keys:
                    backval = pred_dic[backkey]
                    if tval[1] == backval[1]: # 正解の終了位置と同じ終了位置の予測が有る
                        assert backkey != tkey
                        if backkey - frontval[1] <= 2: # 2つの予測の間が1単語以下
                            oracle["merge_span"] += 1
                            true_dic.pop(tkey)
                            pred_dic.pop(tkey)
                            pred_dic.pop(backkey)
                            sentence_ids["merge_span"].append(json_["sentenceID"]+"_"+str(pkey))
                            # 正解 +1
                            # predの予測数 -1
                            break
        
        # split span
        pred_keys = list(pred_dic.keys())
        for pkey in pred_keys:
            pval = pred_dic[pkey]
            if pkey in true_dic: # 予測と同じ開始位置の正解が有る
                frontval = true_dic[pkey]
                true_keys = list(true_dic.keys())
                for backkey in true_keys:
                    backval = true_dic[backkey]
                    if pval[1] == backval[1]: # 予測と同じ終了位置の正解がある
                        assert backkey != pkey
                        if backkey - frontval[1] <= 2: # 2つの予測の間が1単語以下
                            oracle["split_span"] += 1
                            pred_dic.pop(pkey)
                            true_dic.pop(pkey)
                            true_dic.pop(backkey)
                            sentence_ids["split_span"].append(json_["sentenceID"]+"_"+str(pkey))
                            # 正解 +2
                            # predの予測数 +1
                            break
        
        # fix border 
        pred_keys = list(pred_dic.keys())
        for pkey in pred_keys:
            pval = pred_dic[pkey]
            true_keys = list(true_dic.keys())
            for tkey in true_keys:
                tval = true_dic[tkey]
                if pval[0] == tval[0]: # 正解の意味役割と予測の意味役割が一致
                    if (pkey <= tkey and tval[1] <= pval[1]) and sum(true_sentence[pkey:pval[1]+1]) == sum(true_sentence[tkey:tval[1]+1]): # 正解をオーバーラップしている
                        oracle["fix_border"] += 1
                        pred_dic.pop(pkey)
                        true_dic.pop(tkey)
                        sentence_ids["fix_border"].append(json_["sentenceID"]+"_"+str(pkey))
                        # 正解 +1
                        break
        
        # remove arg
        pred_keys = list(pred_dic.keys())
        for pkey in pred_keys:
            pval = pred_dic[pkey]
            if sum(true_sentence[pkey:pval[1]+1]) == 0: # 正解の範囲と重複のない予測を削除
                oracle["remove_arg"] += 1
                pred_dic.pop(pkey)
                sentence_ids["remove_arg"].append(json_["sentenceID"]+"_"+str(pkey))
                # predの予測数 -1
        
        # add arg
        true_keys = list(true_dic.keys())
        for tkey in true_keys:
            tval = true_dic[tkey]
            if sum(pred_sentence[tkey:tval[1]+1]) == 0: # 予測の範囲と重複のない正解を追加
                oracle["add_arg"] += 1
                true_dic.pop(tkey)
                sentence_ids["add_arg"].append(json_["sentenceID"]+"_"+str(tkey))
                # 正解 +1
                # pred + 1
        
        for pd in pred_dic:
            sentence_ids["other_pred"].append(json_["sentenceID"]+"_"+str(pd))
        
        for pd in true_dic:
            sentence_ids["other_ans"].append(json_["sentenceID"]+"_"+str(pd))

        # for key in keys:
        #     sum_ += oracle[key]
        #     oracle[key] = sum_
        # {"fix_labels":0,"move_arg":0,"merge_span":0,"split_span":0,"fix_border":0,"remove_arg":0,"add_arg":0}
        true_num = json_["match_count"]
        pred_num = json_["predict_num"]
        #print(true_num, pred_num, json_["args_num"])

        # all_oracle["fix_labels"][0] += true_num + oracle["fix_labels"]
        # all_oracle["fix_labels"][1] += pred_num
        true_num += oracle["fix_labels"]
        all_oracle_f1["fix_labels"] += cal_f1(true_num,pred_num, json_["args_num"])


        # all_oracle["move_arg"][0] += true_num + oracle["move_arg"]
        # all_oracle["move_arg"][1] += pred_num
        true_num += oracle["move_arg"]
        all_oracle_f1["move_arg"] += cal_f1(true_num,pred_num, json_["args_num"])

        # all_oracle["merge_span"][0] += true_num  + oracle["merge_span"]
        # all_oracle["merge_span"][1] += pred_num - oracle["merge_span"]
        true_num += oracle["merge_span"]
        pred_num -= oracle["merge_span"]
        all_oracle_f1["merge_span"] += cal_f1(true_num,pred_num, json_["args_num"])

        # all_oracle["split_span"][0] += true_num + oracle["split_span"]*2
        # all_oracle["split_span"][1] += pred_num + oracle["split_span"]
        true_num += oracle["split_span"]*2
        pred_num += oracle["split_span"]
        all_oracle_f1["split_span"] += cal_f1(true_num,pred_num, json_["args_num"])

        # all_oracle["fix_border"][0] += true_num + oracle["fix_border"]
        # all_oracle["fix_border"][1] += pred_num
        true_num += oracle["fix_border"]
        all_oracle_f1["fix_border"] += cal_f1(true_num,pred_num, json_["args_num"])

        # all_oracle["remove_arg"][0] += true_num
        # all_oracle["remove_arg"][1] += pred_num - oracle["remove_arg"]
        pred_num -= oracle["remove_arg"]
        all_oracle_f1["remove_arg"] += cal_f1(true_num,pred_num, json_["args_num"])

        # all_oracle["add_arg"][0] += true_num + oracle["add_arg"]
        # all_oracle["add_arg"][1] += pred_num + oracle["add_arg"]
        true_num += oracle["add_arg"]
        pred_num += oracle["add_arg"]
        all_oracle_f1["add_arg"] += cal_f1(true_num,pred_num, json_["args_num"])
    
    keys = ["fix_labels","move_arg","merge_span","split_span","fix_border","remove_arg","add_arg"]
    values = []
    #print(all_oracle_f1, len(test_json_list))
    for key in keys:
        print(key)
        # precision = all_oracle[key][0]*100 / all_oracle[key][1]
        # recall = all_oracle[key][0]*100 / answer_count
        # f1 = 2*precision*recall / (precision+recall)
        # print(f1)
        # print(precision)
        # print(recall)
        f1 = all_oracle_f1[key]/len(test_json_list)
        print(f1)
        values.append(f1)
        # print("precision:\t", precision)
        # print("recall:\t\t", recall)

    with open("error_sentenceid.json", mode="w") as f:
        json.dump(sentence_ids, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main()


