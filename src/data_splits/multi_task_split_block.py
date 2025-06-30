from src.data_splits.base_split_block import data_split
from src.llms import AiBox
from chonkie import NeuralChunker


def init():
    aibox = AiBox(mode='api', model='qw2')
    chunker = NeuralChunker(
        model="mirth/chonky_modernbert_base_1",  # 默认模型
        device_map="cuda:1",  # 运行模型的设备 ('cpu', 'cuda', 等)
        min_characters_per_chunk=10,  # 分块的最小字符数
        return_type="chunks"  # 输出类型
    )
    return aibox,chunker

def compensation_data_split(all_infos:list):
    aibox, chunker = init()
    return data_split(all_infos,aibox,chunker,rule='赔付&领取规则')

def insurance_data_split(all_infos:list):
    aibox, chunker = init()
    return data_split(all_infos,aibox,chunker,rule='投保条款')


if __name__  == '__main__':
    pass