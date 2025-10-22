'''
文本一致性比对agent
'''
from tqdm import tqdm
from loguru import logger
import pandas as pd
import os
import glob, os, re
import json
from typing import List, Dict, Tuple
import concurrent.futures


from src.agents.BaseAgent import BaseAgent
from src.support.prompts import build_text_comp_prompt
from src.support.llms import AiBox
from src.support.qwen_rule_comparison import QwenRuleComparison


model_configs = [
    {
        'model_name': "Qwen/Qwen3-14B",
        'api_url': "https://api.siliconflow.cn/v1/chat/completions",
        'api_key': "sk-ursnzkoxueshqqdzzzlbyujafxgzzobhgesewlpzobqmmobe"
    },
    {
        'model_name': "doubao-1-5-lite-32k-250115",
        'api_url': "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        'api_key': "502f1f26-41e3-45ab-98fb-b31adce50975"
    },
    {
        'model_name': "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        'api_url': "https://api.siliconflow.cn/v1/chat/completions",
        'api_key': "sk-ursnzkoxueshqqdzzzlbyujafxgzzobhgesewlpzobqmmobe"
    }
]

my_rules = ['责任免除', '基础产品销售信息', '投保条款', '保障责任', '保障相关时间', '赔付 & 领取规则',
            '续保条款', '退保条款', '出险条款', '附加条款', '术语解释']

class RuleComparisonMultiVotingAgent(BaseAgent):
    '''
    负责多模态头片的文本比对的agent
    input: 输入待比对的文本对列表，以及对应的rule
    return(bool): 比对的结果：True:文本一致；False: 文本冲突
    example:
        >>> aibox:AiBox = ''
        >>> data_compor = RuleComparisonAgent(aibox)
        >>> data_compor.run(pairs, rule)
    '''
    def __init__(self, aibox:AiBox, use_local_comp_model=False):
        '''
        use_local: 是否使用本地对比模型
        '''
        self.models = [AiBox(config) for config in model_configs]

        if aibox is None or use_local_comp_model:
            qwen_compor = QwenRuleComparison()
            self.llm_inference = qwen_compor.chat
        else:
            self.llm_inference = aibox.chat

    def data_split_block(self, texts: List[str], rule: str) -> List[Tuple[str, str]]:
        if len(texts) < 2:
            return []
        pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                pairs.append((texts[i], texts[j]))
        return pairs



    def _sample_comparison(self, rule: str, text1: str, text2: str) -> str:
        Comparison_System, PT_Text_Comparison = build_text_comp_prompt(rule, text1, text2)
        return self.llm_inference(prompt=PT_Text_Comparison, system=Comparison_System)


    def _res_check(self, result_str: str) -> bool:
        if '无需比较' in result_str: return True
        if '文本冲突' in result_str: return False
        if '文本一致' in result_str: return True
        return True # 这个地方不能false和-1 效果会差

    def majority_vote_overall_result(self, model_final_votes: List[bool]) -> bool:
        true_count = sum(1 for vote in model_final_votes if vote is True)
        false_count = sum(1 for vote in model_final_votes if vote is False)
        if true_count == false_count:
            return True  # 默认平局为无冲突
        return true_count > false_count

    def _process_pairs_for_single_model(self, model: AiBox, rule: str, spilt_blocks: List[Tuple[str, str]]) -> bool:
        """
        辅助函数：单个模型处理所有文本对。
        如果发现冲突（result_int == 0），立即返回 False。
        如果所有对都没有冲突，返回 True。
        """
        for pair_idx, pair in enumerate(spilt_blocks):
            text1, text2 = pair[0], pair[1]
            logger.info(f"    模型 {model.config['model_name']} 正在比较第 {pair_idx + 1} 对文本...")

            system_prompt, user_prompt = build_text_comp_prompt(rule, text1, text2)
            res = model.chat(prompt=user_prompt, system=system_prompt)
            result_int = self._res_check(res)

            logger.info(f"      模型 {model.config['model_name']} 对比结果 (Int: {result_int})：{res[:60]}...")

            # 如果模型返回 0 (文本冲突)，立即停止并返回 False
            if result_int == 0:
                logger.info(
                    f"      模型 {model.config['model_name']} 在第 {pair_idx + 1} 对文本中发现冲突。停止该模型对本材料的后续比对。")
                return False  # 该模型对这个 material 整体判断为【有冲突】

        # 如果所有文本对都处理完毕，且没有发现冲突
        logger.info(f"      模型 {model.config['model_name']} 处理完所有文本对，未发现冲突。")
        return True  # 该模型对这个 material 整体判断为【无冲突】

    # 修正后的 process_model_for_material 函数，用于多线程调用
    def _process_model_for_material_with_timeout(self, model: AiBox, rule: str, spilt_blocks: List[Tuple[str, str]]) -> bool:
        """
        为单个模型处理一个材料下所有文本对的任务设置超时。
        """
        try:
            # 使用 max_workers=1 确保每个模型在自己的线程中独立运行 _process_pairs_for_single_model
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._process_pairs_for_single_model, model, rule, spilt_blocks)
                # 设置一个更长的超时，比如300秒（5分钟），因为这包括了多个 pair 的处理和可能的重试
                return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            logger.error(f"模型 {model.config['model_name']} 处理一个材料的所有文本对超时。")
            # ⚠️ 注意：这里根据你的最新要求，超时被视为无冲突 (True)。
            # 如果你想更谨慎，应改为 False (有冲突)。
            return True  # 超时视为无冲突
        except Exception as e:
            logger.error(f"模型 {model.config['model_name']} 处理材料时发生未预期异常: {e}")
            return False  # 其他异常通常应该被视为有冲突，以防漏报


    def agent_chain(self, df:pd.DataFrame, save_path_json:str, M_DIR:str):
        all_results_for_df = []
        SAVE_PATH_CSV = save_path_json.replace('.jsonl', '.csv')
        os.makedirs(os.path.dirname(save_path_json), exist_ok=True)

        with open(save_path_json, 'a', encoding='utf-8') as jsonl_file:
            for cnt, row in df.iterrows():
                rule, rule_id, material_id = row['rule'], row['rule_id'], row['material_id']

                logger.info(
                    f"\n===============正在处理记录: {cnt} || 材料ID: {material_id} || 规则: {rule}===============")

                current_row_results = {
                    "material_id": material_id,
                    "rule_id": rule_id,
                    "rule": rule,
                    "model1_vote": None,
                    "model2_vote": None,
                    "model3_vote": None,
                    "final_result": None
                }

                if rule not in my_rules:
                    logger.info(f"跳过材料 {material_id} 的规则 {rule} (不在指定规则列表中)。")
                    final_result_for_material = True
                    model_overall_results = [True] * len(self.models)
                    current_row_results.update({f"model{i + 1}_vote": True for i in range(len(self.models))})
                    current_row_results["final_result"] = final_result_for_material
                    jsonl_file.write(json.dumps({
                        "material_id": material_id,
                        "rule_id": rule_id,
                        "result": final_result_for_material,
                        "model_votes": model_overall_results
                    }) + '\n')
                    jsonl_file.flush()
                    all_results_for_df.append(current_row_results)
                    continue

                material_path = os.path.join(M_DIR, material_id)
                module_content_list = []

                if os.path.exists(material_path):
                    for sub_dir in os.listdir(material_path):
                        file_path = os.path.join(material_path, sub_dir, f'{rule}.txt')
                        logger.info(f"尝试从 {file_path} 加载数据...")
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    sample_content = f.read()
                                cleaned_content = sample_content.replace('"', '').replace('\n', '')
                                if len(cleaned_content) < 6:
                                    logger.info(f"    跳过文件 {file_path} (内容长度小于6)。")
                                    continue
                                module_content_list.append(sample_content)
                            except Exception as e:
                                logger.error(f"    读取文件 {file_path} 错误: {e}")
                        else:
                            logger.info(f"    文件未找到: {file_path}")
                else:
                    logger.info(f"    材料目录未找到: {material_path}")

                spilt_blocks = self.data_split_block(module_content_list, rule)

                if len(module_content_list) <= 1 or not spilt_blocks:
                    final_result_for_material = True
                    logger.info(f"    没有足够的文本块进行比较。结果: {final_result_for_material}")
                    model_overall_results = [True] * len(self.models)
                    current_row_results.update({f"model{i + 1}_vote": True for i in range(len(self.models))})
                    current_row_results["final_result"] = final_result_for_material
                    jsonl_file.write(json.dumps({
                        "material_id": material_id,
                        "rule_id": rule_id,
                        "result": final_result_for_material,
                        "model_votes": model_overall_results
                    }) + '\n')
                    jsonl_file.flush()
                    all_results_for_df.append(current_row_results)
                    continue

                # --- 核心并行处理逻辑 ---
                # 使用 ThreadPoolExecutor 并行调用每个模型对当前 material_id 的所有文本对进行评估
                model_overall_results = [None] * len(self.models)  # 用于存储每个模型的最终判断 (True/False)
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                    future_to_model_idx = {
                        executor.submit(self._process_model_for_material_with_timeout, model, rule, spilt_blocks): i
                        for i, model in enumerate(self.models)
                    }
                    for future in concurrent.futures.as_completed(future_to_model_idx):
                        model_idx = future_to_model_idx[future]
                        try:
                            # process_model_for_material_with_timeout 返回 True (无冲突) 或 False (有冲突)
                            model_overall_results[model_idx] = future.result()
                        except Exception as exc:
                            logger.error(f"模型 {self.models[model_idx].config['model_name']} 整体评估时发生异常: {exc}")
                            model_overall_results[model_idx] = False  # 任何意外异常都视为有冲突

                # --- 多数投票确定最终结果 ---
                final_result_for_material = self.majority_vote_overall_result(model_overall_results)

                logger.info(f"    各模型整体评估结果: {model_overall_results}")
                logger.info(f"    材料 {material_id} - 规则 {rule} 的最终多数投票结果: {final_result_for_material}")

                # 更新当前行的结果，准备添加到 DataFrame
                current_row_results.update({
                    f"model{i + 1}_vote": model_overall_results[i] for i in range(len(self.models))
                })
                current_row_results["final_result"] = final_result_for_material
                all_results_for_df.append(current_row_results)

                # 实时写入 JSONL 文件
                jsonl_file.write(json.dumps({
                    "material_id": material_id,
                    "rule_id": rule_id,
                    "result": final_result_for_material,
                    "model_votes": model_overall_results
                }) + '\n')
                jsonl_file.flush()  # 强制写入磁盘

        results_df = pd.DataFrame(all_results_for_df)
        results_df.to_csv(SAVE_PATH_CSV, index=False)
        logger.info(f"\n完成！最终结果已保存至: {SAVE_PATH_CSV}")
        return results_df


    def run(self, df:pd.DataFrame, save_path_json:str, M_DIR:str) -> None:
        self.agent_chain(df, save_path_json, M_DIR)





