from disclaimer_split_block import disclaimer_data_split

data_split_map = {
    '责任免除': disclaimer_data_split
}


def data_split_block(all_infos:list, rule:str):
    return data_split_map[rule](all_infos)



__all__ = [data_split_block]
