'''
数据分块agent
'''
import os
import logging

logger = logging.getLogger(__name__)

from src.agents.BaseAgent import BaseAgent
from src.data_splits import data_split_block
from src.support.word_map_tools import word_map

class RuleInfoSplitAgent(BaseAgent):
    '''负责具体rule数据的分块，输出为文本对列表，提供给文本一致性比较agent进行比对
    input: 输入m文件路径，具体的条款，输出该条款的数据分块
    return(list): [(pair11, pair12), (pair21, pair22), ...]
    example:
        >>> material_path, rule = '',''
        >>> data_spliter = RuleInfoSplitAgent()
        >>> data_spliter.run(material_path, rule)
    '''
    def __init__(self):
        self.tables = str.maketrans(word_map)


    def run(self, material_path:str, rule:str) -> list:
        '''
        将数据进行分块
        '''
        module_content_list = []
        module_file_name_list = []
        for file in os.listdir(material_path):
            path = f'{material_path}/{file}/{rule}.txt'
            logger.info(f"load data {path}...")
            if not os.path.exists(path): continue

            sample = open(path, 'r', encoding='utf-8').read()
            if len(sample.replace('"', '').replace('\n', '')) < 6: continue

            sample = sample.translate(self.tables)

            module_content_list.append(sample)
            module_file_name_list.append(file)

        if len(module_content_list) <=1:
            return []

        return data_split_block(module_content_list, rule)



