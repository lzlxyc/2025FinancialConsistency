'''
分块文本比对
'''
import pandas as pd
import os
import re

from mylogger import setup_logger
logger = setup_logger('../logs/clean_data_保障责任v4.log')

from llms import AiBox
from utils import rule_clauses, get_rule, save_sample
from prompts import Rule_PT_Map, PT_comparison
from data_splits import data_split_block



def sample_comparison(rule:str, text1:str, text2:str, aibox: AiBox):
    rule_des = f"“{rule}”（{rule_clauses[rule]}）"
    RulePT = Rule_PT_Map.get(rule,'')

    Comparison_System = f'''
    你是一个专业的金融保险行业信息处理专家，需要对下面两份文本片段进行冲突分析，不同文本段的部分关键信息可能会被多处定义，售卖平台需要保证这些定义的一致性，
    {RulePT}这就要求对产品物料、售卖素材等进行严格的一致性校验，从而满足监管要求，同时保障客户的合法权益。\n已知：{rule_des}
    '''
    if rule in ['基础产品销售信息']:
        PT_Text_Comparison = (
            f"请对两份片段中的相同条款下的“{rule}”进行比对分析，判断是否存在相互冲突、明显不一致的情况。\n"
            f"需严格区分以下三种情况：\n"
            f"\t1. 【直接冲突】- 当两个片段明确表述不同/相反时\n"
            f"\t2. 【补充说明】- 当一个片段详细说明，另一个片段未提及但可兼容时\n"
            f"\t3. 【信息缺失】- 当某个片段明显缺少必要信息时\n"
            f"核心重点：比对规则：\n"
            f"\t- 仅当文本明确矛盾（直接冲突）时才标记为冲突\n"
            f"\t- 未明确说明不等于否定，表示无冲突\n"
            f"\t- 详细条款与概括性表述不构成冲突，缺少必要信息也表示无冲突。\n"
            f"请开始你的分析：\n\n【片段1】：\n{text1}\n\n【片段2】\n{text2}\n\n"
            f"1) 如果同个保障责任存在冲突，就输出：<res>文本冲突</res>\n<冲突文本段>：\n"
            f"2) 如果不存在冲突，就输出：<res>文本一致</res>\n<冲突文本段>：无。不要有多余信息。")
    elif rule in ['保障责任']:
        PT_Text_Comparison = (
            f"【任务】请对两份片段中的相同条款下的“{rule}”进行比对分析，判断是否存在相互冲突、明显不一致的情况。\n"
            f"【规则】需严格区分以下三种情况：\n"
            f"\t1. 【直接冲突】- 当两个片段明确表述不同/相反时, 标记为冲突\n"
            f"\t2. 【补充说明】- 当一个片段详细说明，另一个片段未提及但可兼容时，表示无冲突\n"
            f"\t3. 【信息缺失】- 当某个片段缺少必要信息时，无法比对，表示无冲突\n"
            f"请开始你的分析：\n\n【片段1】：\n{text1}\n\n【片段2】\n{text2}\n\n"
            f"【输出】\n"
            f"1) 当某个片段缺少必要信息时，就输出：<res>无法比对</res>\n<冲突文本段>：无\n"
            f"2) 如果同个保障责任存在冲突，就输出：<res>文本冲突</res>\n<冲突文本段>：\n"
            f"3) 如果不存在冲突，就输出：<res>文本一致</res>\n<冲突文本段>：无。不要有多余信息。")

    else:
        PT_Text_Comparison = (f"请对两份片段中的相同条款下的“{rule}”进行比对分析，判断是否存在相互冲突、明显不一致的情况。\n"
                              f"相同情形下条款的数量差异不纳入冲突范围,只对比同个条款的差异。\n"
                              f"请开始你的分析：\n\n【片段1】：\n{text1}\n\n【片段2】\n{text2}\n\n"
                              f"1) 如果存在冲突，就输出：<res>文本冲突</res>\n<冲突文本段>：\n"
                              f"1) 如果不存在冲突，就输出：<res>文本一致</res>\n<冲突文本段>：无。不要有多余信息。")


    return aibox.chat(prompt=PT_Text_Comparison, system=Comparison_System)


def text_comparison_main(data_name:str='验证集', nrows=None):
    '''文本一致性匹配'''
    def check(result_str: str) -> bool:
        if '文本冲突' in result_str: return False
        if '文本一致' in result_str: return True

        return -1

    def check(result_str: str) -> bool:
        if rule == '保障责任':
            if '无法比较' in result_str: return -1
            if '文本一致' in result_str: return True
            if '文本冲突' in result_str:
                if (lines:=result_str.split('<冲突文本段>：')[-1].split('\n')):
                    lines = [line for line in lines if line and not any(k in line for k in ['未提及','未明确','未具体'])]
                    # print(f'{lines=}')
                    if lines: return False

                else: return -1
            return -1

        else:
            if '文本冲突' in result_str: return False
            if '文本一致' in result_str: return True
            return -1

    aibox = AiBox(mode='api',model='qw2')
    M_DIR = f'../DATA/{data_name}/materials'
    SAVE_PATH = f'../DATA/{data_name}/clean_data_保障责任_splitv4.jsonl'
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)

    df = pd.read_json(f"../DATA/{data_name}/data.jsonl", lines=True)

    df_sample = pd.read_json(f"../../2025FinancialConsistency/outputs/submit0.8701.jsonl", lines=True)
    assert len(df) == len(df_sample)

    ypreds = []
    if nrows is not None:
        df = df.head(nrows)

    df['rule'] = df['rule'].apply(get_rule)
    print('***count::', len(df))

    cnt = -1
    for row in df.iloc[:].iterrows():
        cnt += 1
        rule, rule_id, material_id = row[1].rule, row[1].rule_id, row[1].material_id
        label = row[1].result if 'result' in df.columns else None

        filter_materials = []

        if material_id in filter_materials or rule not in ['保障责任']:
            material_id = df_sample.iloc[cnt].material_id
            rule_id = df_sample.iloc[cnt].rule_id
            end_result = bool(df_sample.iloc[cnt].result)

            logger.info(f"===============skip cnt：{cnt+1} || {material_id=} || {rule=} || {end_result=}===============")
            ypreds.append(end_result)
            save_sample(SAVE_PATH, material_id, rule_id, end_result)
            continue


        material_path = f'{M_DIR}/{material_id}'

        logger.info(f"\n===============cnt：{cnt} || {material_id=} || {rule=} || {label=}===============")

        module_content_list = []
        module_file_name_list = []
        for file in os.listdir(material_path):
            path = f'{material_path}/{file}/{rule}.txt'
            logger.info("*"*150)
            logger.info(f"load data {path}...")
            if not os.path.exists(path): continue

            sample = open(path, 'r', encoding='utf-8').read()

            sample = re.sub(r'\n+', '\n', sample.replace('"',''))

            if len(sample.replace('\n','')) < 6: continue

            # module_content_list.append(f"\n{material_id} {file}\n")
            module_content_list.append(sample)
            module_file_name_list.append(file)

            logger.info(sample)

        logger.info("开始比对.....................................")
        spilt_blocks = data_split_block(module_content_list, rule)

        for sam in spilt_blocks:
            logger.info(f'\n{sam}')



        if len(module_content_list) <= 1 or not spilt_blocks:
            end_result = True
            logger.info(f"===============<=1 {cnt=} || {material_id=} || {rule=} || {end_result=}===============")
            ypreds.append(end_result)
            save_sample(SAVE_PATH, material_id, rule_id, end_result)
            continue

        results = []
        try:
            for pair in spilt_blocks:
                text1, text2 = pair[0], pair[1]
                sample = sample_comparison(rule, text1, text2, aibox)
                result = check(sample)
                results.append(result)
                logger.info(f"{text1} \nvs\n {text2} \n>>> result={sample} || res:{result}")

                if not result: break

            logger.info(f"results: {results}")
            end_result = all(res for res in results) if results else True

        except Exception as e:
            logger.error(e)
            end_result = True

    #
        logger.info(f"==============={cnt=} || {material_id=} || {rule=} || {end_result=}===============")
        ypreds.append(end_result)
        save_sample(SAVE_PATH, material_id, rule_id, end_result)

    df['ypred'] = ypreds
    df.to_csv(SAVE_PATH.replace('.jsonl', '.csv'), index=False)
    logger.info(f"Done! 文件保存至:{SAVE_PATH.replace('.jsonl', '.csv')}")


if __name__ == '__main__':
    text_comparison_main(data_name='测试A集_clean')