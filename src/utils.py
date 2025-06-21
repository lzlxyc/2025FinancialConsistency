import markdown
import glob, os
from llms import AiBox
from tqdm import tqdm
# from prompts import PT_Extract_RuleInfo

rule_clauses = {
    "基础产品销售信息": "该保险产品的基础配置信息，包括产品名、附加的条款信息、销售限制等",
    "投保条款": "投保过程中的缴费约定、投被保人条件限制等",
    "保障责任": "约定该产品的保险责任细节，如保障范围、保险金额、增值服务等",
    "保障相关时间": "约定该产品的各类时间信息，包括但不限于犹豫期、等待期、宽限期等",
    "赔付 & 领取规则": "约定该产品的保险责任的赔付、给付、领取及免赔细节，如赔付年龄/比例/次数等",
    "责任免除": "约定该产品不承担保险责任的情形,险人不承担给付保险金的责任",
    "续保条款": "约定续保相关信息，包括但不限于续保条件、保证续保等",
    "退保条款": "约定退保相关信息，包括但不限于退保条件、退保手续费等",
    "出险条款": "约定出险相关信息，包括但不限于出险地点、出险方式等",
    "附加条款": "约定该产品的附加条款，如特别约定等",
    "术语解释": "约定该产品的术语解释，如名词定义等"
}



def read_markdown(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        html = markdown.markdown(text)

    return html


def rule_info_extract_form_md(rule:str, md_path:str, aibox:AiBox) -> str:
    '''提取出单个md文件中特定规则的完整数据'''
    rule = f"{rule}（{rule_clauses[rule]}）"
    
    PT_Extract_RuleInfo = f"提取出下面文本的{rule}信息，要完整的、不要有遗漏的信息，更不要修改数据内容。如果相关的本文本不存在，则输出一个空字符串。"
    sample = read_markdown(md_path)
    
    print(PT_Extract_RuleInfo)
    print(sample)
    return aibox.chat(prompt=sample, system=PT_Extract_RuleInfo)

     
def rule_info_extract_from_file(rule:str, md_dir:str, aibox:AiBox) -> None:
    '''提取出一份素材（几个md文件）中特定规则的完整数据'''
    all_infos = []
    for path in glob.glob(f"{md_dir}/*.md"):
        sample_response = rule_info_extract_form_md(rule, path, aibox)
        print(f'processing file:{path}...')
        
        if sample_response:
            all_infos.append(sample_response)

    with open(f'{md_dir}/{rule}.txt', 'w', encoding='utf-8') as fp:
        for sample in all_infos:
            fp.write(sample + '\n\n')
            
    # return all_infos

def rule_info_extract(rule:str, doc_dir: str, aibox:AiBox) -> None:
    '''
    rule: 规则
    doc_dir： materials子文件夹
    '''
    for doc in tqdm(os.listdir(doc_dir)):
        md_dir = doc_dir + '/doc'
        print(f'processing docs:{doc}...')
        rule_info_extract_from_file(rule, md_dir, aibox)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    
    aibox = AiBox(mode='local')
    res = rule_info_extract_form_md('责任免除','../DATA/验证集/materials/m_00002s/LIABILITY_EXCLUSION/0.md',aibox)
    print(res)




    

    
