# from src.data_splits.disclaimer_split_block import disclaimer_data_split
# from src.data_splits.multi_task_split_block import \
#     compensation_data_split,\
#     insurance_data_split,\
#     insurance_time_data_split,\
#     product_sales_information_data_split,\
#     term_explanation_data_split,\
#     indemnity_responsibility_data_split

from .disclaimer_split_block import disclaimer_data_split
from .indemnity_time_split_block import indemnity_time_data_split

from .multi_task_split_block import \
    compensation_data_split,\
    insurance_data_split,\
    insurance_time_data_split,\
    product_sales_information_data_split,\
    term_explanation_data_split,\
    indemnity_responsibility_data_split,\
    surrender_data_split,\
    additional_term_data_split,\
    renewal_term_data_split,\
    claim_clause_data_split

data_split_map = {
    '责任免除': disclaimer_data_split,
    '赔付 & 领取规则': compensation_data_split,
    '投保条款': insurance_data_split,
    '保障相关时间': indemnity_time_data_split,
    '基础产品销售信息': product_sales_information_data_split,
    '术语解释': term_explanation_data_split,
    '保障责任': indemnity_responsibility_data_split,
    '退保条款': surrender_data_split,
    '附加条款': additional_term_data_split,
    '续保条款': renewal_term_data_split,
    '出险条款': claim_clause_data_split,
}


def data_split_block(all_infos:list, rule:str):
    return data_split_map[rule](all_infos)


__all__ = [data_split_block]
