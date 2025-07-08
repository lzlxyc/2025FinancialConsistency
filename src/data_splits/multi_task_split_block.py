from src.data_splits.base_split_block import data_split
from src.llms import AiBox
# from chonkie import NeuralChunker



# def init():
#     aibox = AiBox(mode='api', model='qw2')
#     chunker = NeuralChunker(
#         model="mirth/chonky_modernbert_base_1",  # 默认模型
#         device_map="cuda:1",  # 运行模型的设备 ('cpu', 'cuda', 等)
#         min_characters_per_chunk=10,  # 分块的最小字符数
#         return_type="chunks"  # 输出类型
#     )
#     return aibox,chunker

def init():
    aibox = AiBox(mode='api', model='qw2')
    # chunker = NeuralChunker(
    #     model="mirth/chonky_modernbert_base_1",  # 默认模型
    #     device_map="cuda:1",  # 运行模型的设备 ('cpu', 'cuda', 等)
    #     min_characters_per_chunk=10,  # 分块的最小字符数
    #     return_type="chunks"  # 输出类型
    # )
    return aibox,[]


def compensation_data_split(all_infos:list):
    aibox, chunker = init()
    return data_split(all_infos,aibox,chunker,rule='赔付&领取规则')


def insurance_data_split(all_infos:list):
    try:
        from src.data_splits.insurance_data_split_block import insurance_data_split
        return insurance_data_split(all_infos)
    except:
        print('无实例化，使用大模型分块')
        aibox, chunker = init()
        return data_split(all_infos,aibox,chunker,rule='投保条款')


def insurance_time_data_split(all_infos:list):
    aibox, chunker = init()
    return data_split(all_infos,aibox,chunker,rule='保障相关时间')


def product_sales_information_data_split(all_infos:list):
    aibox, chunker = init()
    return data_split(all_infos, aibox, chunker, rule='基础产品销售信息')

def term_explanation_data_split(all_infos:list):
    aibox, chunker = init()
    return data_split(all_infos, aibox, chunker, rule='术语解释')

def indemnity_responsibility_data_split(all_infos:list):
    from src.data_splits.indemnity_responsibility_split_block import indemnity_responsibility_data_split
    aibox, _ = init()
    return indemnity_responsibility_data_split(all_infos,aibox)


if __name__  == '__main__':
    pass