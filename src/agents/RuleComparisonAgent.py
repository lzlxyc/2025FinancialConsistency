'''
文本一致性比对agent
'''
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor


from src.agents.BaseAgent import BaseAgent
from src.support.prompts import build_text_comp_prompt
from src.support.llms import AiBox
from src.support.qwen_rule_comparison import QwenRuleComparison

logger = logging.getLogger(__name__)

class RuleComparisonAgent(BaseAgent):
    '''
    负责进行文本比对的agent
    input: 输入待比对的文本对列表，以及对应的rule
    return(bool): 比对的结果：True:文本一致；False: 文本冲突
    example:
        >>> aibox:AiBox = ''
        >>> data_compor = RuleComparisonAgent(aibox)
        >>> data_compor.run(pairs, rule)
    '''
    def __init__(self, aibox:AiBox, use_local_comp_model=False):
        '''
        use_local: 是否使用本地对比模型
        '''
        if aibox is None or use_local_comp_model:
            qwen_compor = QwenRuleComparison()
            self.llm_inference = qwen_compor.chat
        else:
            self.llm_inference = aibox.chat



    def _sample_comparison(self, rule: str, pair:tuple[str, str]) -> str:
        text1, text2 = pair[0], pair[1]
        Comparison_System, PT_Text_Comparison = build_text_comp_prompt(rule, text1, text2)

        sample = self.llm_inference(prompt=PT_Text_Comparison, system=Comparison_System)

        if True or '文本冲突' in sample:
        # if '无需比较' in sample:
            logger.info(f"{text1} \nvs\n {text2} \n>>> result={sample}")

        # 后处理
        return self._res_check(sample)


    def _res_check(self, result_str: str) -> bool:
        if '无需比较' in result_str: return True
        if '文本冲突' in result_str: return False
        if '文本一致' in result_str: return True
        return True # 这个地方不能false和-1 效果会差


    def run(self, pairs_to_comp:list, rule:str) -> bool:
        if not pairs_to_comp:
            return True

        # results = []
        # for pair in tqdm(pairs_to_comp):
        #     result = self._sample_comparison(rule, pair)
        #     results.append(result)
        #     if not result:
        #         break

        all_results = []

        thread_num = 6
        for i in tqdm(range(0, len(pairs_to_comp), thread_num)):
            pairs = pairs_to_comp[i:i+thread_num]

            with ThreadPoolExecutor(max_workers=thread_num) as executor:
                # 使用map简化代码
                results = executor.map(
                    lambda pair: self._sample_comparison(rule, pair),
                    pairs
                )
                tmp_res = list(results)
                all_results += tmp_res
                if any(res is False for res in tmp_res):
                    break

        logger.info(f"results: {all_results}")
        return all(res for res in all_results) if all_results else True
