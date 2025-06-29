'''
文本比对
'''
import pandas as pd
import os

from mylogger import setup_logger
logger = setup_logger('../logs/clean_data_责任免除v2.log')

from llms import AiBox
from utils import rule_clauses, get_rule, save_sample

from prompts import Rule_PT_Map

from src.data_splits.tools import spilt_block_by_socre

def sample_comparison(rule:str, text1:str, text2:str, aibox: AiBox):
    rule_des = f"“{rule}”（{rule_clauses[rule]}）"
    RulePT = Rule_PT_Map.get(rule,'')

    Comparison_System = f"你是一个专业的金融保险行业信息处理专家，请对下面两份文本片段进行冲突分析，最终输出两份片段是否有冲突。\n已知：{rule_des}"
    PT_Text_Comparison = (f"请对两份片段中的相同情形下的“{rule}”进行比对分析，判断是否存在相互冲突、明显不一致的情况。"
                          f"如果遇到冲突的情况就直接返回结果，不必继续往下分析。"
                          f"请开始你的分析：\n\n【片段1】：\n{text1}\n\n【片段2】\n{text2}\n\n"
                          f"请按照下面的格式输出："
                          f"<think>分析</think>>"
                          f"最终结果：<result>冲突/不冲突</result>")


    Comparison_System = '''
    你是一个专业的金融保险行业信息处理专家，需要对下面两份文本片段进行冲突分析，不同文本段的部分关键信息可能会被多处定义，售卖平台需要保证这些定义的一致性，
    以某责任的赔付比例为例，如果条款、投保须知中定义的赔付比例都为80%，而售卖介绍图片中将比例错配为90%，那么用户在理赔时便可能会产生纠纷、诉讼风险。
    这就要求对产品物料、售卖素材等进行严格的一致性校验，从而满足监管要求，同时保障客户的合法权益。\n已知：{rule_des}
    '''
    PT_Text_Comparison = (f"请对两份片段中的相同条款下的“{rule}”进行比对分析，判断是否存在相互冲突、明显不一致的情况。\n"
                          f"相同情形下条款的数量差异不纳入冲突范围,只对比同个条款的差异\n"
                          f"如果遇到冲突的情况就直接返回结果，不必继续往下分析。"
                          f"请开始你的分析：\n\n【片段1】：\n{text1}\n\n【片段2】\n{text2}\n\n"
                          f"并按照下面的格式输出："
                          f"<think>分析</think>>"
                          f"最终结果：<result>冲突/不冲突</result>")

    Comparison_System = f'''
    你是一个专业的金融保险行业信息处理专家，需要对下面两份文本片段进行冲突分析，不同文本段的部分关键信息可能会被多处定义，售卖平台需要保证这些定义的一致性，
    以某责任的赔付比例为例，如果免责条款中定义的有效条件是“被保险人感染艾滋病病毒（HIV）或患艾滋病（AIDS）期间”，而售卖介绍图片中将条款生效条件定义为“被保险人感染艾滋病病毒（HIV）或患艾滋病（AIDS）、高血压III级期间；”，那么用户在理赔时便可能会产生纠纷、诉讼风险。
    这就要求对产品物料、售卖素材等进行严格的一致性校验，从而满足监管要求，同时保障客户的合法权益。\n已知：{rule_des}
    '''
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

    aibox = AiBox(mode='api',model='qw2')
    M_DIR = f'../DATA/{data_name}/materials'
    SAVE_PATH = f'../DATA/{data_name}/clean_data_责任免除_splitv2.jsonl'
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)

    df = pd.read_json(f"../DATA/{data_name}/data.jsonl", lines=True)

    df_sample = pd.read_json(f"../outputs/submit0.76.jsonl", lines=True)
    assert len(df) == len(df_sample)

    # mids = ['m_00002s', 'm_00005s', 'm_000014s', 'm_000020s', 'm_00001a', 'm_00003a','m_00004a','m_00007a','m_00139a','m_00061a']
    # df = df[df['material_id'].isin(mids)]
    print(df.head())

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

        filter_materials = ['m_00007a', 'm_00038a', 'm_00060a', 'm_00108a', 'm_00128a']
        # mids = ['m_00003a', 'm_00006a', 'm_00024a', 'm_00027a', 'm_00055a','m_00059a', 'm_00082a', 'm_00119a', 'm_00041a', 'm_00116a','m_00133a']


        if material_id in filter_materials or rule != '责任免除':
            material_id = df_sample.iloc[cnt].material_id
            rule_id = df_sample.iloc[cnt].rule_id
            end_result = bool(df_sample.iloc[cnt].result)

            logger.info(f"===============skip cnt：{cnt+1} || {material_id=} || {rule=} || {end_result=}===============")
            ypreds.append(end_result)
            save_sample(SAVE_PATH, material_id, rule_id, end_result)
            continue


        material_path = f'{M_DIR}/{material_id}'

        logger.info(f"===============cnt：{cnt} || {material_id=} || {rule=} || {label=}===============")

        module_content_list = []
        module_file_name_list = []
        for file in os.listdir(material_path):
            path = f'{material_path}/{file}/{rule}.txt'
            logger.info(f"load data {path}...")
            if not os.path.exists(path): continue

            sample = open(path, 'r', encoding='utf-8').read()
            if len(sample.replace('"','').replace('\n','')) < 6: continue

            # module_content_list.append(f"\n{material_id} {file}\n")
            module_content_list.append(sample)
            module_file_name_list.append(file)

        spilt_blocks = spilt_block_by_socre(module_content_list, rule)

        if len(module_content_list) <= 1 or not spilt_blocks:
            end_result = False
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
                logger.info(f"{text1} \nvs\n {text2} \n>>> result={sample}")
                # logger.info(f"{result} >> {text1}\n{text2}\n\n")

                if not result: break

            logger.info(f"results: {results}")
            end_result = all(res for res in results) if results else True

        except Exception as e:
            logger.error(e)
            end_result = False

    #
        logger.info(f"==============={cnt=} || {material_id=} || {rule=} || {end_result=}===============")
        ypreds.append(end_result)
        save_sample(SAVE_PATH, material_id, rule_id, end_result)

    df['ypred'] = ypreds
    df.to_csv(SAVE_PATH.replace('.jsonl', '.csv'), index=False)
    logger.info(f"Done! 文件保存至:{SAVE_PATH.replace('.jsonl', '.csv')}")


if __name__ == '__main__':
    text_comparison_main(data_name='测试A集_clean')