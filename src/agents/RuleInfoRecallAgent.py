'''
负责条款协议的召回agent
'''
import os, glob
import logging

from src.agents.BaseAgent import BaseAgent
from src.support.llms import AiBox
from src.support.utils import read_markdown
from src.support.prompts import built_recall_system_pt
from src.support.qwen_rule_standard import QwenRuleStandard

logger = logging.getLogger(__name__)

class RuleInfoRecallAgent(BaseAgent):
    '''负责具体rule数据的召回
    input: 输入m文件路径，具体的条款，输出该条款的数据分块
    return(str): 召回的文本
    example:
        >>> aibox:AiBox = ''
        >>> data_recallor = RuleInfoRecallAgent(aibox)
        >>> data_recallor.run(material_path, rule)
    '''
    def __init__(self, aibox:AiBox, recall_mode='model', is_rule_pre_standard=False) -> None:
        '''
        recall_mode: 召回模式：regular(规则、关键词检索)、model（大模型）、mix（混合模式）
        is_rule_pre_standard: 表示是否需要进行数据标准化
        '''
        self.aibox = aibox
        self.recall_mode = recall_mode
        self.qwen_rule_standard  = QwenRuleStandard() if is_rule_pre_standard else None

    def _rule_info_extract_form_md(self, rule: str, md_path: str, chunk_size: int = 50000) -> str:
        '''分段循环抽取规则信息并拼接，避免输入超长报错'''
        system_prompt = built_recall_system_pt(rule)

        text = read_markdown(md_path)
        results = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            res = ""
            for attempt in range(3):
                try:
                    res = self.aibox.chat(prompt=chunk, system=system_prompt)
                    break  # 成功则跳出重试循环
                except Exception as e:
                    if attempt < 2:
                        import time
                        time.sleep(10)
                    else:
                        print(f"chunk [{start}:{end}] all retries failed: {e}")
            if res.strip():
                results.append(res.strip())
            start = end

        return '\n'.join(results)


    def _rule_info_extract_from_file(self, rule: str, md_dir: str) -> None:
        '''提取出一份素材（几个md文件）中特定规则的完整数据'''
        save_path = f'{md_dir}/{rule}.txt'
        if os.path.exists(save_path):
            print(f'file {save_path} exists......')
            return

        all_infos = []
        for path in glob.glob(f"{md_dir}/*.md"):
            print(f'processing file:{path}...')
            sample_response = self._rule_info_extract_form_md(rule, path)
            print('*' * 200)
            # print(sample_response)

            if sample_response:
                # 进行排版标准化
                if self.qwen_rule_standard is not None:
                    sample_response = self.qwen_rule_standard.chat(sample_response)

                all_infos.append(sample_response)

        with open(save_path, 'w', encoding='utf-8') as fp:
            for sample in all_infos:
                fp.write(sample + '\n\n')

        # return all_infos


    def _rule_info_extract(self, rule: str, doc_dir: str) -> None:
        '''
        rule: 规则
        doc_dir： materials子文件夹
        '''
        for doc in os.listdir(doc_dir):
            md_dir = doc_dir + f'/{doc}'
            print(f'processing docs:{doc}...')
            self._rule_info_extract_from_file(rule, md_dir)


    def run(self, material_path:str, rule:str) -> None:
        '''数据召回外部调用接口'''
        print(f'processing material_path:{material_path}<<====>>{rule}...')
        self._rule_info_extract(rule, material_path)