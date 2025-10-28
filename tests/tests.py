import os
import sys
import argparse
from dataclasses import dataclass
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import text_validate

##############################################
#  recall_mode: 召回模式：regular(规则、关键词检索)、model（大模型）、mix（混合模式）
#  data_split_mode: 数据分块模式：regular(正则)、model（神经网络模型）、mix（混合模式）
#  compare_mode: 文本比对模式：single(单模型模式)、ensemble(多模式模式)、train_model(微调模型模式)
##############################################


@dataclass
class ConfigTests:
    recall_mode: str    # 召回方式
    data_split_mode:str # 分块方式
    compare_mode:str    # 对比方式

    log_file:str
    save_file:str
    data_name = 'test_datas'
    model_mode = 'api'
    model = 'qw32'

    is_rule_pre_standard = False


def main_test():
    '''进行消融测试
    召回：召回1(关键词+语义检索)、召回2(大模型)、召回3(混合检索)
    分块：分块1（纯正则分块）、分块2（纯神经网络分块）、分块3（混合分块）
    对比：对比1(单模型模式)、对比2(多模式模式)、对比3(微调模型模式)
    recall_modes = ['regular','model','mix']
    data_split_modes = ['regula','model','mix']
    compare_modes = ['single', 'ensemble','train_model']
    '''
    recall_modes = ['model']
    data_split_modes = ['mix']
    compare_modes = ['ensemble']

    all_metrics = []
    for recall in recall_modes:
        for data_split in data_split_modes:
            for compare in compare_modes:
                curr_mode = '_'.join([recall, data_split, compare]) + '_ab'
                log_file = f'./logs/{curr_mode}.log'
                save_file = curr_mode
                print(f'------------------------{curr_mode}-------------------------')
                args = ConfigTests(
                    recall_mode=recall,
                    data_split_mode=data_split,
                    compare_mode=compare,
                    save_file=save_file,
                    log_file=log_file
                )
                print(vars(args))

                metrics = text_validate(args)
                metrics['mode'] = curr_mode
                all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv('./reports/all_metrics.csv', index=False)


if __name__ == '__main__':
    main_test()
