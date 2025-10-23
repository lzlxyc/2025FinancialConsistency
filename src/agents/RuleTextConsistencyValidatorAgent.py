'''
分块文本比对
'''
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

logger = logging.getLogger(__name__)

my_rule = [
    '赔付 & 领取规则',
    '责任免除',
    '投保条款',
    '保障相关时间',
    '保障责任',
    '基础产品销售信息',
    '退保条款',
    '附加条款',
    '续保条款',
    '出险条款',
    '术语解释',
]
from src.agents.BaseAgent import BaseAgent
from src.support.llms import AiBox
from src.support.utils import load_data, save_sample

from .RuleInfoRecallAgent import RuleInfoRecallAgent
from .RuleInfoSplitAgent import RuleInfoSplitAgent
from .RuleComparisonAgent import RuleComparisonAgent
from .RuleComparisonMultiVotingAgent import RuleComparisonMultiVotingAgent

class RuleTextConsistencyValidatorAgent(BaseAgent):
    '''
    负责对输入文件夹中的条款文本一致性对比检验
    example:
        >>> data_name = ''
        >>> comp_agent = RuleTextConsistencyValidatorAgent(data_name)
        >>> comp_agent.run()
    '''
    def __init__(self, data_name:str, model_mode='api',
                 model='qw72',api_key={},save_file='result',
                 use_local_comp_model=False,
                 is_rule_pre_standard=False,
                 is_use_voting_model=False):
        '''
        data_name: 数据集名称
        use_local_comp_model：是否使用本地模型进行比对
        '''
        self.data_name = data_name
        self.aibox = AiBox(mode=model_mode, model=model, api_key=api_key)
        self.M_DIR = f'data/{data_name}/materials'
        self.SAVE_PATH = f'reports/{save_file}.jsonl'
        if os.path.exists(self.SAVE_PATH):
            os.remove(self.SAVE_PATH)

        self.is_use_voting_model = is_use_voting_model

        self.rule_info_recall_agent = RuleInfoRecallAgent(self.aibox, is_rule_pre_standard)
        self.rule_info_split_agent = RuleInfoSplitAgent()

        if is_use_voting_model:
            self.rule_comparison_agent = RuleComparisonMultiVotingAgent()
        else:
            self.rule_comparison_agent = RuleComparisonAgent(self.aibox, use_local_comp_model)

        self.test_datas = None

    def agent_chain(self):
        df, df_sample = load_data(self.data_name)
        # df = df.iloc[:3]
        ypreds = []

        for cnt, row in enumerate(df.iterrows()):
            rule, rule_id, material_id, ytrue = (row[1].rule,
                                                 row[1].rule_id,
                                                 row[1].material_id,
                                                 row[1].result)
            material_path = f'{self.M_DIR}/{material_id}'

            if rule in my_rule:
                # 召回
                self.rule_info_recall_agent.run(material_path, rule)
                # 分块
                pairs_to_comp = self.rule_info_split_agent.run(material_path, rule)
                # 对比
                end_result = self.rule_comparison_agent.run(pairs_to_comp, rule)
                logger_info = f"=========={len(pairs_to_comp)} "
            else:
                end_result = bool(df_sample.iloc[cnt].result)
                logger_info = f"===========skip "

            res = bool(end_result == ytrue)
            _info = f"{logger_info}:{cnt=}|{material_id=}|{rule=}|{end_result=}|{ytrue=}|{res=}==============="
            logger.info(_info)

            ypreds.append(end_result)
            save_sample(self.SAVE_PATH, material_id, rule_id, end_result, ytrue)

        df['ypred'] = ypreds
        df.to_csv(self.SAVE_PATH.replace('.jsonl', '.csv'), index=False)
        logger.info(f"Done! 文件保存至:{self.SAVE_PATH.replace('.jsonl', '.csv')}")

        self.test_datas = df


    def run(self):
        self.agent_chain()
        logger.info("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def compute_metrics(self):
        '''进行指标的统计：
        acc, p, r, f1
        '''
        logger.info(">>>>>>>>> 进数据的统计分析：")
        if self.test_datas is not None:
            # 获取真实标签和预测标签
            y_true = self.test_datas['result']
            y_pred = self.test_datas['ypred']

            # 计算各项指标
            acc = accuracy_score(y_true, y_pred)
            p = precision_score(y_true, y_pred, average='binary')  # 二分类
            r = recall_score(y_true, y_pred, average='binary')  # 二分类
            f1 = f1_score(y_true, y_pred, average='binary')  # 二分类

            # 打印结果
            metric_info = (f"\n====================================\n"
                           f"准确率 (Accuracy): {acc*100:.2f}%\n"
                           f"精确率 (Precision): {p*100:.2f}%\n"
                           f"召回率 (Recall): {r*100:.2f}%\n"
                           f"F1分数 (F1-score): {f1*100:.2f}%\n"
                           f"====================================\n")
            logger.info(metric_info)

            # 返回指标字典
            metrics = {
                'accuracy': acc,
                'precision': p,
                'recall': r,
                'f1': f1
            }

            return metrics
        else:
            print("测试数据为空，无法计算指标")
            return None

