from src.data_splits.base_split_block import data_split,data_no_split
from src.support.llms import AiBox
from chonkie import NeuralChunker



def init():
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    chunker = NeuralChunker(
        model="D:/LZL/workspace/ModelHub/chonky_modernbert_base_1",  # 默认模型
        device_map="cuda:0",  # 运行模型的设备 ('cpu', 'cuda', 等)
        min_characters_per_chunk=1,  # 分块的最小字符数
        return_type="chunks",
    )
    chunker.model.eval()

    return chunker

chunker = init()

def compensation_data_split(all_infos:list):# 赔付&领取规则
    return  data_split(all_infos, chunker, rule='赔付&领取规则')

def insurance_data_split(all_infos:list):# 投保条款
    return data_no_split(all_infos,)


def insurance_time_data_split(all_infos:list):#保障相关时间
    return data_no_split(all_infos,)


def product_sales_information_data_split(all_infos:list):#基础产品销售信息
    return data_no_split(all_infos,)

def indemnity_responsibility_data_split(all_infos:list): # 没有提升
    return data_no_split(all_infos,)

def disclaimer_data_split(all_infos: list):# 责任免除
    from disclaimer_split_block import disclaimer_data_split
    return disclaimer_data_split(all_infos)

def surrender_data_split(all_infos:list):# 退保条款
    return data_no_split(all_infos,)

def additional_term_data_split(all_infos:list):# 附加条款
    return data_no_split(all_infos,)

def renewal_term_data_split(all_infos:list):#续保条款
    return data_no_split(all_infos,)

def claim_clause_data_split(all_infos:list):#出险条款
    return data_no_split(all_infos,)

def term_explanation_data_split(all_infos:list):#术语解释
    return data_no_split(all_infos,)


if __name__  == '__main__':
    pass