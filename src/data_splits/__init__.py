from src.data_splits.disclaimer_split_block import disclaimer_data_split
from src.data_splits.compensation_split_block import compensation_data_split

data_split_map = {
    '责任免除': disclaimer_data_split,
    '赔付 & 领取规则': compensation_data_split

}


def data_split_block(all_infos:list, rule:str):
    return data_split_map[rule](all_infos)



__all__ = [data_split_block]
