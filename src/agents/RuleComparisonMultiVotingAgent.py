'''
文本一致性比对agent
'''
from tqdm import tqdm
import logging
from typing import List
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.BaseAgent import BaseAgent
from src.support.prompts import build_text_comp_prompt
from src.support.llms import AiBox
from src.support.qwen_rule_comparison import QwenRuleComparison

logger = logging.getLogger(__name__)


model_configs = [
    {
        'model': "deepseek-chat",
        'api_url': "https://api.deepseek.com",
        'api_key': "sk-c11b4bd9dadc4e41ad6ae6dccdbbfd6e"
    },
    {
        'model': "qwen2.5-72b-instruct",
        'api_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'api_key': "sk-cb027866cd8a41568be43a0d8fde952d"
    },
    {
        'model': "qwen2.5-32b-instruct",
        'api_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'api_key': "sk-cb027866cd8a41568be43a0d8fde952d"
    },
]


model_configs = [
    {
        'model': "DeepSeek-v3.2-Exp",
        'api_url': "https://api.poe.com/v1",
        'api_key': "6o2pEkzlRNq0nZPsPVQFw2kTPmS9jzN8HrBs4u21_8E"
    },
    {
        'model': "qwen2.5-72b-instruct",
        'api_url': "https://api.poe.com/v1",
        'api_key': "6o2pEkzlRNq0nZPsPVQFw2kTPmS9jzN8HrBs4u21_8E"
    },
    {
        'model': "qwen2.5-32b-instruct",
        'api_url': "https://api.poe.com/v1",
        'api_key': "6o2pEkzlRNq0nZPsPVQFw2kTPmS9jzN8HrBs4u21_8E"
    },
]

class RuleComparisonMultiVotingAgent(BaseAgent):
    '''
    多模型负责进行文本比对的agent
    input: 输入待比对的文本对列表，以及对应的rule
    return(bool): 比对的结果：True:文本一致；False: 文本冲突
    '''
    def __init__(self):
        '''
        use_local: 是否使用本地对比模型
        '''
        self.ai_boxs = [AiBox(**config) for config in model_configs]
        print("进行集成策略，使用模型：", [ai.model for ai in self.ai_boxs])

    def _sample_comparison(self, rule: str, pair:tuple[str, str], ai_box:AiBox) -> str:
        text1, text2 = pair[0], pair[1]
        Comparison_System, PT_Text_Comparison = build_text_comp_prompt(rule, text1, text2)
        sample = ai_box.chat(prompt=PT_Text_Comparison, system=Comparison_System)
        if '文本冲突' in sample:
            logger.info(f"{text1} \nvs\n {text2} \n>>> result={sample}")
        # 后处理
        return self._res_check(sample)


    def _res_check(self, result_str: str) -> bool:
        if '无需比较' in result_str: return True
        if '文本冲突' in result_str: return False
        if '文本一致' in result_str: return True
        return True # 这个地方不能false和-1 效果会差

        # 新增：每个ai_box的处理逻辑（供线程池调用）

    def _process_ai_box(self, ai_box, pairs_to_comp, rule):
        results = []
        for pair in pairs_to_comp:
            result = self._sample_comparison(rule, pair, ai_box)
            results.append(result)
            if not result:  # 一旦出现冲突，提前终止该模型的后续比对
                break
        logger.info(f"ai_box results: {results}")
        return all(results) if results else True

    def run(self, pairs_to_comp:list, rule:str) -> bool:
        if not pairs_to_comp:
            return True

        s_time = time()
        all_res = []
        # 使用线程池并行处理多个ai_box
        with ThreadPoolExecutor(max_workers=len(self.ai_boxs)) as executor:
            # 提交所有ai_box的任务
            futures = {
                executor.submit(self._process_ai_box, ai_box, pairs_to_comp, rule): ai_box
                for ai_box in self.ai_boxs
            }

            # 实时获取结果（按完成顺序）
            for future in tqdm(as_completed(futures), total=len(futures), desc="多模型并行处理"):
                try:
                    res = future.result()  # 获取该ai_box的处理结果
                    all_res.append(res)
                except Exception as e:
                    logger.error(f"ai_box处理出错: {str(e)}")
                    # 出错时可按业务需求处理（这里默认视为"一致"）
                    all_res.append(True)

        # 进行投票（原逻辑保持不变）
        logger.info(f">>>>>>>>>>>> 多模型投票：all results: {all_res} ****{round(time() - s_time, 4)}")

        true_count = sum(all_res)
        false_count = len(all_res) - true_count

        return true_count > false_count


