from src.data_splits.base_split_block import data_split
from src.llms import AiBox
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
    aibox = AiBox(mode='api', model='qw3')
    chunker = NeuralChunker(
        model="mirth/chonky_modernbert_base_1",  # 默认模型
        device_map="cuda:1",  # 运行模型的设备 ('cpu', 'cuda', 等)
        min_characters_per_chunk=1,  # 分块的最小字符数
        return_type="chunks",
    )
    chunker.model.eval()

    return aibox,chunker
aibox, chunker = init()
# def init():
#     aibox = AiBox(mode='api', model='qw2')
#     # chunker = NeuralChunker(
#     #     model="mirth/chonky_modernbert_base_1",  # 默认模型
#     #     device_map="cuda:1",  # 运行模型的设备 ('cpu', 'cuda', 等)
#     #     min_characters_per_chunk=10,  # 分块的最小字符数
#     #     return_type="chunks"  # 输出类型
#     # )
#     return aibox,[]


# def compensation_data_split(all_infos:list):
#     return data_split(all_infos,aibox,chunker,rule='赔付&领取规则')

def compensation_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='赔付&领取规则')

def insurance_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='投保条款')


def insurance_time_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='保障相关时间')


def product_sales_information_data_split(all_infos:list):
    # return data_split(all_infos, aibox, chunker, rule='基础产品销售信息')
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='基础产品销售信息')

def indemnity_responsibility_data_split(all_infos:list):
    # from src.data_splits.indemnity_responsibility_split_block import indemnity_responsibility_data_split
    # aibox, _ = init()
    # return indemnity_responsibility_data_split(all_infos,aibox)
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='保障责任')

def disclaimer_data_split_v2(all_infos: list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='责任免除')

def surrender_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos,aibox,chunker,rule='退保条款')

def additional_term_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos,aibox,chunker,rule='附加条款')

def renewal_term_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos,aibox,chunker,rule='续保条款')

def claim_clause_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos,aibox,chunker,rule='出险条款')

def term_explanation_data_split(all_infos:list):
    from src.data_splits.surrender_data_split_block import surrender_data_split
    return surrender_data_split(all_infos, aibox, chunker, rule='术语解释')

if __name__  == '__main__':
    pass