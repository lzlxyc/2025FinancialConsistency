'''
负责条款协议的召回agent
'''
import os, re, glob
import logging
from concurrent.futures import ThreadPoolExecutor


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
    def __init__(self,
                 aibox:AiBox,
                 recall_mode='model',
                 data_pre_process=False,
                 is_rule_pre_standard=False) -> None:
        '''
        recall_mode: 召回模式：regular(规则、关键词检索)、model（大模型）、mix（混合模式）
        data_pre_process: 是否使用数据预处理
        is_rule_pre_standard: 表示是否需要进行数据标准化
        '''
        self.aibox = aibox
        self.recall_mode = recall_mode
        self.data_pre_process = data_pre_process
        self.qwen_rule_standard  = QwenRuleStandard() if is_rule_pre_standard else None


    def base_preprocess_to_block(self, text:str, chunk_size=50000) -> list:
        blocks = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            blocks.append(chunk)
            start = end

        return blocks

    def data_preprocess_to_block(self, data: str) -> list:
        '''加上?=断言:return:
        '''
        for pattern in [
            r'第[一二三四五六七八九十]+部分\s',
            r'\n\b\d+	',
            r'\n\b\d+\．\s'
        ]:
            if len(re.findall(pattern, data)) >= 3:
                break

        # print(pattern)
        all_blocks = []

        if '部分' in pattern:
            big_split_datas = re.split(f'(?={pattern})|(?=\n.+（互联网.+条款)', data)
            # if len(max(big_split_datas, key=len)) > 10000:
            for split_data in big_split_datas:
                if split_data:
                    all_blocks += re.split(r'(?=第[一二三四五六七八九十]+条[\n\t\s])', split_data)
            all_blocks = [d for d in all_blocks if d is not None]

        else:
            all_patterns = r'(?=\n.+（互联网专属）条款)|(?=\n[一二三四五六七八九十]+、)|(?=\n【\d+\．[\n\t\s])|(?=第[一二三四五六七八九十]+条\n)|(?=\n第[一二三四五六七八九十]+条[\n\t\s])'
            big_split_datas = re.split(all_patterns, data)
            for split_data in big_split_datas:
                if split_data:
                    all_blocks += re.split(f'(?={pattern})', split_data)

        all_split_blocks = []
        for split_data in all_blocks:
            all_split_blocks += re.split(r'(?=【.*】[\n\t\s])|\n\n\n', split_data)

        if len(max(all_split_blocks, key=len)) > 100000:
            end_all_split_blocks = []
            for split_data in all_split_blocks:
                if len(split_data) <= 10000:
                    end_all_split_blocks.append(split_data)
                else:
                    tmp = re.split(r'(?=\n\d+．\d+\s)|(?=\n\d+\s)|(?=\n（\d+）)', split_data)
                    end_all_split_blocks += tmp

            all_split_blocks = [data for data in end_all_split_blocks if len(data) >= 1]

        return [a for d in all_split_blocks if len(a := re.sub(r'\n+', '\n', d)) >= 4]


    def _rule_info_extract_form_md(self, rule: str, md_path: str, chunk_size: int = 10000) -> str:
        '''分段循环抽取规则信息并拼接，避免输入超长报错'''
        print(f'processing file:{md_path}...')
        text = read_markdown(md_path).replace('.', '．')
        text = re.sub('\n\s', '\n', text)
        if len(text) <= 6: return ''

        if self.data_pre_process:
            try:
                blocks = self.data_preprocess_to_block(text)
                return self.recall(rule, blocks)
            except Exception as e:
                print(">>>>>>>>>>> recall", e)

        blocks = self.base_preprocess_to_block(text, chunk_size)
        return self.recall(rule, blocks)


    def recall(self, rule, blocks):
        system_prompt = built_recall_system_pt(rule)

        with ThreadPoolExecutor(max_workers=6) as executor:
            # 使用map简化代码
            results = executor.map(
                lambda block: self.aibox.chat(prompt=block, system=system_prompt),
                blocks
            )
            recall_res = list(results)
            recall_res = [res for res in recall_res if len(res) >=4]

        # recall_res = []
        # for block in blocks:
        #     res = self.aibox.chat(prompt=block, system=system_prompt)
        #     if len(res) >=4:
        #         recall_res.append(res)

        print('#' * 150)
        sample_response = '\n'.join(recall_res)

        if sample_response and self.qwen_rule_standard is not None:
            sample_response = self.qwen_rule_standard.chat(sample_response)

        return sample_response



    def _rule_info_extract_from_file(self, rule: str, md_dir: str) -> None:
        '''提取出一份素材（几个md文件）中特定规则的完整数据'''
        recall_data_mode = '.preproce.txt' if self.data_pre_process else '.txt'
        save_path = f'{md_dir}/{rule}{recall_data_mode}'
        if os.path.exists(save_path):
            print(f'file {save_path} exists......')
            return

        all_paths = [path for path in glob.glob(f"{md_dir}/*.md")]
        all_infos = []
        for path in all_paths:
            sample_response = self._rule_info_extract_form_md(rule, path)
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