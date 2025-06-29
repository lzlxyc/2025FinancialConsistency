import re

from src.data_splits.tools import (
    zh_same_string,
    diff_similarity,
    ngram_similarity
)

from src.llms import AiBox
from chonkie import NeuralChunker,SemanticChunker,SDPMChunker
from tqdm import tqdm
aibox = AiBox(mode='api',model='qw2')
chunker = NeuralChunker(
        model="mirth/chonky_modernbert_base_1",  # 默认模型
        device_map="cuda:1",  # 运行模型的设备 ('cpu', 'cuda', 等)
        min_characters_per_chunk=10,  # 分块的最小字符数
        return_type="chunks"  # 输出类型
    )
def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def  translate2ENG(text):
    rule = "赔付&领取规则"
    translate2Eng_System = f'''
    你是一个专业的金融保险行业英文翻译专家，需要对下面的金融保险关于{rule}的条款准确翻译为英文并返回。你需要遵循以下规则：
    1.你的输出只有翻译后的英文，不要有任何其他输出，
    2.如果输入为''，直接返回null。
    '''
    prompt = f'''
    请对以下关于{rule}的条款进行准确翻译：{text}
    '''
    return aibox.chat(prompt=prompt, system=translate2Eng_System)

def __compensation_data_presplit(data:str) -> list:
    text = data
    if re.search(r'[\u4e00-\u9fff]', text) is None:
        return []
    clean_text = remove_blank_line(text)
    lines = clean_text.splitlines()  # 每行为一个中文条款句子

    # 遍历并翻译为英文
    translated_lines = []
    for line in tqdm(lines):
        eng = line
        for attempt in range(3):
            try:
                eng = translate2ENG(line).strip()
                break
            except Exception as e:
                print(f"翻译失败，第 {attempt + 1} 次尝试: {e}")
        translated_lines.append(eng)
        # print(line)
        # print(eng)
    text_eng = "\n".join(translated_lines)
    chunks = chunker.chunk(text_eng)
    eng2zh_map = list(zip(translated_lines, lines))

    blocks = []
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        # print(chunk_text)
        # 收集 chunk 中涉及到的中文原文
        chunk_zh = []
        for eng_line, zh_line in eng2zh_map:
            if eng_line in chunk_text:
                chunk_zh.append(zh_line)
        blocks.append("\n".join(chunk_zh))  # block 中是该chunk对应的中文内容
    return blocks
    #    print('==============================')
    # for i in blocks:
    #     print(i)
    #     print('----------------------')

def remove_blank_line(text: str) -> str:
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_blank_lines)

def compensation_data_split(all_infos:list):
    '''赔偿分块'''
    inputs = []
    for info in all_infos:
        blocks = __compensation_data_presplit(info)
        blocks = [s for s in blocks if len(s.split()) > 1 or len(s) >= 50]
        blocks = list(dict.fromkeys(blocks))
        inputs.append(blocks)

    sim_blocks = []
    len_inputs = len(inputs)
    for i in range(len_inputs):
        for j in range(i+1, len_inputs):
            blocks_i = inputs[i]
            blocks_j = inputs[j]
            for bi in blocks_i:
                for bj in blocks_j:
                    if zh_same_string(bi, bj): continue
                    score = diff_similarity(bi, bj)
                    n_score = ngram_similarity(bi, bj)
                    if score > 0.75 or n_score > 0.8:
                        sim_blocks.append((bi, bj))

    return sim_blocks




def test_chonkie():

    text = """2.5 保险责任

在本合同保险期间内，且本合同有效的前提下，我们按以下约定承担保险责任：

2.5.1 恶性肿瘤医疗保险金

被保险人在等待期届满后，在我们指定医疗机构具有合法资质的专科医生初次确诊患有本合同约定的恶性肿瘤，在我们指定医疗机构接受治疗的，对被保险人因治疗恶性肿瘤而发生的符合通常惯例的、医学必须且合理的医疗费用，我们对下述 2.5.1.1-2.5.1.4 类费用，按照本合同的约定承担给付恶性肿瘤医疗保险金的责任：

2.5.1.1 恶性肿瘤住院医疗费用

被保险人住院期间发生的应当由被保险人支付的、必需且合理的住院医疗费用，包括床位费、膳食费、护理费、重症监护室床位费、诊疗费、检查检验费、治疗费、药品费、手术费。如果在本合同约定的保险期间届满之日，被保险人仍未结束本次住院治疗的，对于自本合同保险期间届满之日起 30 日内（含第 30 日）因本次住院治疗发生的必需且合理的住院医疗费用，我们继续承担保险责任。

2.5.1.2 恶性肿瘤特殊门诊医疗费用

被保险人在门诊（不含特需门诊）接受下述特殊治疗期间发生的应当由被保险人支付的、必需且合理的如下特殊门诊医疗费用：化学疗法、放射疗法、肿瘤免疫疗法、肿瘤内分泌疗法、肿瘤靶向疗法治疗费用。

2.5.1.3 恶性肿瘤门诊手术医疗费用

被保险人接受门诊手术治疗期间发生的应当由被保险人支付的、必需且合理的治疗恶性肿瘤门诊手术费用。

2.5.1.4 恶性肿瘤住院前后门急诊医疗费用

在本合同约定的保险期间内，被保险人在住院前 7 日（含住院当日）和出院后 30 日（含出院当日）内，因与本次住院相同原因而接受恶性肿瘤门急诊治疗的，被保险人在前述医疗机构接受门急诊治疗期间发生的应当由被保险人支付的、必需且合理的治疗恶性肿瘤门急诊医疗费用（但不包括本合同第 2.5.1.2 和 2.5.1.3 项约定的恶性肿瘤特殊门诊医疗费用和恶性肿瘤门诊手术医疗费用）。

我们对于以上四类费用的累计赔偿金额以本合同约定的恶性肿瘤医疗保险金的保险金额为限，一次或累计赔偿的金额达到保险单载明的恶性肿瘤医疗保险金额时，我们对被保险人在恶性肿瘤医疗保险金项下的保险责任终止。

2.7 免赔额

本合同中所指免赔额均为年免赔额，指一个保单年度内，应由被保险人自行承担，本合同不予赔付的部分。本合同中恶性肿瘤医疗保险金免赔额，由您和我们协商确定并在保险单中载明。以下可以计入年免赔额的范围：

（1）被保险人从其它商业性费用补偿型医疗保险获得的医疗费用补偿；
（2）除社会医疗保险和公费医疗保障以外，被保险人从其他途径获得的医疗费用补偿。

注：被保险人通过社会医疗保险和公费医疗保障获得的补偿，不可用于抵扣免赔额。

2.8 补偿原则和赔付标准

本合同适用医疗费用补偿原则。我们按如下约定给付恶性肿瘤医疗保险金：

1. 若被保险人未从社会医疗保险、公费医疗、其它商业性费用补偿型医疗保险、其他政府机构或者社会福利机构、其他责任方获得医疗费用补偿，我们按如下公式根据本合同的约定给付恶性肿瘤医疗保险金：
   恶性肿瘤医疗保险金=（被保险人实际支出的符合本合同相关约定的医疗费用-免赔额）×赔付比例
   免赔额及赔付比例在保单中载明，累计给付金额以保险单载明的相应保险金额为限。

2. 若被保险人已从社会医疗保险、公费医疗、其它商业性费用补偿型医疗保险、其他政府机构或者社会福利机构、其他责任方获得医疗费用补偿（以下简称已获得的医疗费用补偿），我们按如下公式根据本合同的约定给付恶性肿瘤医疗保险金：
   恶性肿瘤医疗保险金=（被保险人实际支出的符合本合同相关约定的医疗费用-已获得的医疗费用补偿-免赔额）×赔付比例
   保险金额、免赔额及赔付比例在保险单中载明，且该赔付比例应高于前述未从社会医疗保险等途径获得补偿时的赔付比例。

3. 社保卡的个人账户支出部分视为个人支付，不属于已获得的医疗费用补偿。

4. 被保险人以参加社会医疗保险身份投保，但未以参加社会医疗保险身份就诊并结算或结算金额为 0 的，我们按如下公式根据本合同的约定给付恶性肿瘤医疗保险金：
   恶性肿瘤医疗保险金=（被保险人实际支出的符合本合同相关约定的医疗费用-已获得的医疗费用补偿-免赔额）×赔付比例。
   免赔额及赔付比例在保单中载明，累计给付金额以保险单载明的相应保险金额为限。未足额补缴当期保险费的，则本合同自动终止。若您未按照约定支付分期保费，且本合同终止前发生保险事故的，我们扣减欠缴的保险费后按照本合同约定承担保险责任；对于本合同终止后发生的保险事故，我们不承担保险责任。"""
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("mirth/chonky_modernbert_base_1",use_auth_token=True,)
    from chonkie import LateChunker, RecursiveRules
#     text="""
#     2.5 Insurance Liability.
# During the insurance period of this contract and on the premise that this contract is valid, we shall assume the insurance liability as follows:
# 2.5.1 Medical insurance benefits for Malignant tumors
# After the expiration of the waiting period, if the insured is initially diagnosed with a malignant tumor as stipulated in this contract by a specialist doctor with legal qualifications at the medical institution designated by us and receives treatment at the medical institution designated by us, for the medical expenses incurred by the insured due to the treatment of the malignant tumor that are in line with common practice, medically necessary and reasonable, We assume the responsibility of paying the medical insurance benefit for malignant tumors in accordance with the provisions of this contract for the following types of expenses ranging from 2.5.1.1 to 2.5.1.4:
# 2.5.1.1 Hospitalization medical expenses for malignant tumors
# The necessary and reasonable hospitalization medical expenses that should be paid by the insured during the insured's hospitalization period, including bed fees, meal fees, nursing fees, intensive care unit bed fees, consultation fees, examination and test fees, treatment fees, drug fees, and operation fees. If the insured has not completed the current hospitalization treatment by the expiration date of the insurance period stipulated in this contract, we will continue to assume the insurance liability for the necessary and reasonable hospitalization medical expenses incurred due to this hospitalization treatment within 30 days (including the 30th day) from the expiration date of the insurance period stipulated in this contract.
# 2.5.1.2 Special outpatient medical expenses for malignant tumors
# The following special outpatient medical expenses that should be paid by the insured during the period when the insured receives the following special treatments at the outpatient department (excluding special outpatient services), which are necessary and reasonable: treatment expenses for chemotherapy, radiotherapy, tumor immunotherapy, tumor endocrine therapy, and tumor targeted therapy.
# 2.5.1.3 Medical expenses for outpatient surgery of malignant tumors
# The necessary and reasonable outpatient surgical expenses for the treatment of malignant tumors that should be paid by the insured during the period when the insured receives outpatient surgical treatment.
# 2.5.1.4 Outpatient and emergency medical expenses before and after hospitalization for malignant tumors
# During the insurance period stipulated in this contract, if the insured receives outpatient or emergency treatment for malignant tumors for the same reason as this hospitalization within 7 days before hospitalization (including the day of hospitalization) and 30 days after discharge (including the day of discharge), The necessary and reasonable medical expenses for the treatment of malignant tumors that should be paid by the insured during the period when the insured receives outpatient and emergency treatment at the aforementioned medical institutions (but excluding the special outpatient medical expenses for malignant tumors and outpatient surgical medical expenses for malignant tumors as stipulated in items 2.5.1.2 and 2.5.1.3 of this contract).
# The cumulative compensation amount for the above four types of expenses shall be limited to the insurance amount of the malignant tumor medical insurance as stipulated in this contract. When the one-time or cumulative compensation amount reaches the malignant tumor medical insurance amount stated in the insurance policy, our insurance liability for the insured under the malignant tumor medical insurance shall terminate.
# 2.7 Deductible
# The deductibles referred to in this contract are all annual deductibles, which refer to the portion that should be borne by the insured themselves and not covered by this contract within one policy year. The deductible for the medical insurance for malignant tumors in this contract shall be determined through negotiation between you and us and stated in the insurance policy. The following can be included in the annual deductible range:
# (1) Medical expense compensation obtained by the insured from other commercial cost-compensating medical insurances;
# (2) Compensation for medical expenses obtained by the insured from other sources except for social medical insurance and public medical security.
# The compensation obtained by the insured through social medical insurance and public medical security cannot be used to offset the deductible.
# 2.8 Compensation Principles and Standards
# This contract is subject to the principle of medical expense compensation. We pay the medical insurance benefit for malignant tumors as follows:
# 1. If the insured has not received medical expense compensation from social medical insurance, public medical care, other commercial expense compensation medical insurance, other government agencies or social welfare institutions, or other responsible parties, we will pay the malignant tumor medical insurance benefit in accordance with the following formula as stipulated in this contract:
# The medical insurance benefit for malignant tumors = (the actual medical expenses incurred by the insured in accordance with the relevant provisions of this contract - the deductible) × the compensation ratio
# The deductible and the compensation ratio are stated in the policy. The cumulative payment amount is limited to the corresponding insurance amount stated in the policy.
# 2. If the insured has received medical expense compensation from social medical insurance, public medical care, other commercial expense compensation medical insurance, other government agencies or social welfare institutions, or other responsible parties (hereinafter referred to as the medical expense compensation already obtained), we will pay the malignant tumor medical insurance benefit in accordance with the following formula as stipulated in this contract:
# The medical insurance benefit for malignant tumors = (the actual medical expenses incurred by the insured in accordance with the relevant provisions of this contract - the medical expense compensation already received - the deductible) × the compensation ratio
# The insurance amount, deductible and compensation ratio are stated in the insurance policy, and such compensation ratio should be higher than the compensation ratio when no compensation is obtained from social medical insurance or other channels as mentioned above.
# 3. The personal account expenditure portion of the social security card is regarded as personal payment and does not fall under the medical expense compensation already obtained.
# 4. If the insured insured is insured under the social medical insurance but fails to receive medical treatment and settle accounts under the social medical insurance or the settlement amount is 0, we will pay the malignant tumor medical insurance benefit in accordance with the following formula as stipulated in this contract:
# The medical insurance benefit for malignant tumors = (the actual medical expenses incurred by the insured in accordance with the relevant provisions of this contract - the medical expense compensation already received - the deductible) × the compensation ratio.
# The deductible and the compensation ratio are stated in the policy. The cumulative payment amount is limited to the corresponding insurance amount stated in the policy. If the current insurance premium is not paid in full, this contract will be automatically terminated. If you fail to pay the installment premium as agreed and an insurance incident occurs before the termination of this contract, we will deduct the outstanding insurance premium and assume the insurance liability as stipulated in this contract. We shall not assume insurance liability for any insurance incidents that occur after the termination of this contract.
#     """
    if re.search(r'[\u4e00-\u9fff]', text) is None:
        return
    clean_text = remove_blank_line(text)
    lines = clean_text.splitlines()  # 每行为一个中文条款句子

    # 遍历并翻译为英文
    translated_lines = []
    for line in tqdm(lines):
        eng = line
        for attempt in range(3):
            try:
                eng = translate2ENG(line).strip()
                break
            except Exception as e:
                print(f"翻译失败，第 {attempt + 1} 次尝试: {e}")
        translated_lines.append(eng)
        print(line)
        print(eng)
    text_eng = "\n".join(translated_lines)
    chunks = chunker.chunk(text_eng)
    eng2zh_map = list(zip(translated_lines, lines))

    blocks = []
    for chunk in chunks:
        chunk_text = chunk.text.strip()
        print(chunk_text)
        # 收集 chunk 中涉及到的中文原文
        chunk_zh = []
        for eng_line, zh_line in eng2zh_map:
            if eng_line in chunk_text:
                chunk_zh.append(zh_line)
        blocks.append("\n".join(chunk_zh))  # block 中是该chunk对应的中文内容
        print('==============================')
    for i in blocks:
        print(i)
        print('----------------------')
if __name__  == '__main__':
    test_chonkie()