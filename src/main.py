import os
import argparse
from dotenv import load_dotenv

from src.agents.RuleTextConsistencyValidatorAgent import (
    RuleTextConsistencyValidatorAgent
)
from .logger import setup_logger

os.makedirs('reports', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def get_api_key() -> dict:
    assert os.path.exists('./.env')
    load_dotenv('./.env')

    return {'ds' : os.getenv('DS_API_KEY'),
            'qwen' : os.getenv('QWEN_API_KEY')}


def text_validate(args):
    '''
    recall_mode: 召回模式：regular(规则、关键词检索)、model（大模型）、mix（混合模式）
    data_split_mode: 数据分块模式：regular(正则)、model（神经网络模型）、mix（混合模式）
    compare_mode: 文本比对模式：single(单模型模式)、ensemble(多模式模式)、train_model(微调模型模式)
    '''
    # 配置统一的日志
    setup_logger(args.log_file, True)

    validator = RuleTextConsistencyValidatorAgent(
        data_name=args.data_name,
        model_mode=args.model_mode,
        model=args.model,
        api_key=get_api_key(),
        save_file=args.save_file,
        data_pre_process=args.data_pre_process,
        recall_mode=args.recall_mode,
        data_split_mode=args.data_split_mode,
        compare_mode=args.compare_mode
    )
    validator.run()
    metrics = validator.compute_metrics()

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFAC文本一致性对比系统')

    parser.add_argument('--data_name', type=str, default='test_datas', help='测试集名称')
    parser.add_argument('--model_mode', type=str, default='api', help='调用模型方式')
    parser.add_argument('--model', type=str, default='ds', help='调用大模型具体模型:ds/qw72')
    parser.add_argument('--save_file', type=str, default='result', help='保存的文件名称')
    parser.add_argument('--log_file', type=str, default='./logs/test_data.log', help='日志名称')
    parser.add_argument('--use_local_comp_model', type=bool, default=False, help='是否使用本地比对模型')
    parser.add_argument('--is_rule_pre_standard', type=bool, default=False, help='是否使用大模型进行前期的数据标准化')
    parser.add_argument('--is_use_voting_model', type=bool, default=False, help='是否使用多模型投票策略')

    args = parser.parse_args()
    text_validate(args)
