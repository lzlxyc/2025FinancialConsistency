import re
import pandas as pd
import json

# from prompts import PT_Extract_RuleInfo

def get_rule(rule):
    rule = re.sub('该产品的|在各材料中的定义没有冲突|与', '', rule)
    return re.sub('的时间', '时间', rule)


def get_mid2rule(data_name='验证集'):
    df = pd.read_json(f'data/{data_name}/data.jsonl', lines=True)
    df['rule'] = df['rule'].apply(get_rule)
    df = df.groupby('material_id').agg(tuple).reset_index()
    mid2rule = dict(df[['material_id','rule']].values)


    with open(f'data/{data_name}/mid2rule.json', 'w', encoding='utf-8') as f:
        json.dump(mid2rule, f, ensure_ascii=False, indent=4)

    return mid2rule


def read_markdown(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        # html = markdown.markdown(text)

    return text


def save_sample(
        save_path:str,
        material_id:str,
        rule_id:str,
        result:bool,
        ytrue=None) -> None:
    with open(save_path, "a") as up:
        up.write(json.dumps({
            "material_id": material_id,
            "rule_id": rule_id,
            "ypred": result,
            "ytrue": ytrue
        }) + "\n")


def load_data(data_name):
    df = pd.read_json(f"data/{data_name}/ab.jsonl", lines=True)
    # df_sample = pd.read_json(f"outputs/notrans_['赔付 & 领取规则'].jsonl", lines=True)
    # assert len(df) == len(df_sample)

    df['rule'] = df['rule'].apply(get_rule)
    print('***count::', len(df))

    return df, None



    

    
