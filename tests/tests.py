import os
import sys
import argparse
from dataclasses import dataclass
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import text_validate

@dataclass
class ConfigTests:
    recall: str    # 召回方式
    data_split:str # 分块方式
    compare:str    # 对比方式

    log_file:str
    save_file:str
    data_name = 'test_datas'
    model_mode = 'api'
    model = 'ds'

    use_local_comp_model = False
    is_rule_pre_standard = False
    is_use_voting_model = False


def main_test():
    '''进行消融测试
    召回：召回1(关键词+语义检索)、召回2(llm)、召回3(混合检索)
    分块：分块1（纯正则分块）、分块2（纯神经网络分块）、分块3（混合分块）
    对比：对比1(单模型)、对比2(多模型)、对比3(训练模型)
    recall_modes = ['keywords','llm','mix']
    data_split_modes = ['regula','nn','mix']
    compare_modes = ['simgle', 'more','train']
    '''
    recall_modes = ['llm']
    data_split_modes = ['mix']
    compare_modes = ['simgle']

    all_metrics = []
    for recall in recall_modes:
        for data_split in data_split_modes:
            for compare in compare_modes:
                curr_mode = '_'.join([recall, data_split, compare])
                log_file = f'./logs/{curr_mode}.log'
                save_file = curr_mode
                print(f'------------------------{curr_mode}-------------------------')
                args = ConfigTests(
                    recall=recall,
                    data_split=data_split,
                    compare=compare,
                    save_file=save_file,
                    log_file=log_file
                )
                metrics = text_validate(args)
                metrics['mode'] = curr_mode
                all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv('./reports/all_metrics.csv', index=False)


if __name__ == '__main__':
    main_test()
