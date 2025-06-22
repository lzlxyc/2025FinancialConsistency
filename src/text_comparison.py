'''
文本比对
'''
import pandas as pd
import glob, os

from mylogger import setup_logger
logger = setup_logger('../logs/text_comparison1.log')

from llms import AiBox
from utils import rule_clauses, get_rule, save_sample


def sample_comparison(rule:str, text1:str, text2:str, aibox: AiBox):
    rule_des = f"“{rule}”（{rule_clauses[rule]}）"
    Comparison_System = f"你是一个专业的金融保险行业信息处理专家，请对下面两份文本中的“{rule}”进行严谨的冲突分析，最终输出两份文件是否有冲突。"
    PT_Text_Comparison = (f"1、分析下述两个文件的{rule_des}，对两分文件中都提到的同一件事物的描述和表达，判断是否存在明显相互冲突的情况。\n"
                          f"2、如果对某件事物，其中一份文件未明确说明、或者省略说明时不必进行比较，即表示没有冲突\n"
                          f"3、只比较相同条件下两份文件的描述是否冲突，比如赔偿金额的冲突、保险病种的冲突，并请按照下面的格式输出：\n"
                          f"<thing>具体的内容冲突分析</thing>\n"
                          f"<result>有冲突\无冲突</result>\n\n"
                          f"【文件1】：\n{text1}\n\n【文件2】\n{text2}")


    return aibox.chat(prompt=PT_Text_Comparison, system=Comparison_System)


def text_comparison_main(data_name:str='验证集', nrows=None):
    '''文本一致性匹配'''
    aibox = AiBox(mode='api',model='qw72')
    M_DIR = f'../DATA/{data_name}/materials'
    SAVE_PATH = f'../DATA/{data_name}/sample2.jsonl'

    df = pd.read_json(f"../DATA/{data_name}/data.jsonl", lines=True)

    mids = ['m_00002s', 'm_00005s', 'm_000014s', 'm_000020s', 'm_00001a', 'm_00003a','m_00004a','m_00007a','m_00139a','m_00061a']
    df = df[df['material_id'].isin(mids)]
    print(df.head())

    ypreds = []

    if nrows is not None:
        df = df.tail(nrows)

    df['rule'] = df['rule'].apply(get_rule)

    cnt = 0
    for row in df.iloc[:].iterrows():
        cnt += 1
        rule, rule_id, material_id = row[1].rule, row[1].rule_id, row[1].material_id
        label = row[1].result if 'result' in df.columns else None

        material_path = f'{M_DIR}/{material_id}'

        logger.info(f"==============={cnt=} || {material_id=} || {rule=} || {label=}===============")

        module_content_list = []
        for file in os.listdir(material_path):
            path = f'{material_path}/{file}/{rule}.txt'
            logger.info(f"load data {path}...")
            if not os.path.exists(path): continue

            sample = open(path, 'r', encoding='utf-8').read()
            if len(sample) < 10: continue

            module_content_list.append(sample)

        if len(module_content_list) <= 1:
            end_result = True
            logger.info(f"===============<=1 {cnt=} || {material_id=} || {rule=} || {end_result=}===============")
            ypreds.append(end_result)
            save_sample(SAVE_PATH, material_id, rule_id, end_result)
            continue

        results = []

        try:
            for i in range(len(module_content_list)):
                if i == len(module_content_list) - 1: break

                for j in range(i+1, len(module_content_list)):
                    text1, text2 = module_content_list[i], module_content_list[j]
                    sample = sample_comparison(rule, text1, text2, aibox)
                    sample_res = sample.split('<result>', maxsplit=1)[-1]
                    result = bool('无冲突' in sample_res)
                    results.append(result)
                    logger.info(f"{i}|{j} result={sample}")
                    # logger.info(f"{result} >> {text1}\n{text2}\n\n")

                    if not result: break

                if not results[-1]: break

            logger.info(f"results: {results}")
            end_result = all(res for res in results) if results else True

        except Exception as e:
            print(e)
            end_result = False

        logger.info(f"==============={cnt=} || {material_id=} || {rule=} || {end_result=}===============")
        ypreds.append(end_result)
        save_sample(SAVE_PATH, material_id, rule_id, end_result)

    df['ypred'] = ypreds
    df.to_csv(SAVE_PATH.replace('.jsonl', '.csv'), index=False)
    logger.info(f"Done! 文件保存至:{SAVE_PATH.replace('.jsonl', '.csv')}")


if __name__ == '__main__':
    text_comparison_main(nrows=6)