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
    model:str
    model_mode:str = 'api'
    data_name:str = 'test_datas'

    data_pre_process:bool = False


def main_test():
    '''进行消融测试
    预处理：data_pre_process：是否使用数据预处理： True\False
    召回：召回1(关键词+语义检索)、召回2(大模型)、召回3(混合检索)
    分块：分块1（纯正则分块）、分块2（纯神经网络分块）、分块3（混合分块）
    对比：对比1(单模型模式)、对比2(多模式模式)、对比3(微调模型模式)
    data_pre_process = [True, False]
    recall_modes = ['regular','model','mix']
    data_split_modes = ['regula','model','mix']
    compare_modes = ['single', 'ensemble','train_model']
    '''
    model = 'qw32'
    data_pre_process = [True]
    recall_modes = ['model']
    data_split_modes = ['mix']
    compare_modes = ['single']

    all_metrics = []
    for data_preprocess_mode in data_pre_process:
        for recall in recall_modes:
            for data_split in data_split_modes:
                for compare in compare_modes:
                    curr_mode = '_'.join([recall, data_split, compare, model])
                    log_file = f'./logs/{curr_mode}.log'
                    save_file = curr_mode
                    print(f'------------------------{curr_mode}-------------------------')
                    args = ConfigTests(
                        recall_mode=recall,
                        data_split_mode=data_split,
                        compare_mode=compare,
                        model=model,
                        save_file=save_file,
                        log_file=log_file,
                        data_pre_process=data_preprocess_mode,
                    )
                    print(vars(args))

                    metrics = text_validate(args)
                    metrics['mode'] = curr_mode
                    all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv('./reports/all_metrics.csv', index=False)


if __name__ == '__main__':
    main_test()
