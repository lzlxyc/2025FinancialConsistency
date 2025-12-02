import pandas as pd
import re
import markdown
import glob, os, re
from tqdm import tqdm
import pandas as pd
import json
from openai import OpenAI



def read_final_block(file_path):
    '''读取 final_block.txt 文件内容并返回 JSON 数据'''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        return json.loads(content)  # 尝试直接解析 JSON 对象，返回的是一个列表[]

    except json.JSONDecodeError:
        print(f"Failed to decode JSON in file: {file_path}.")
        return []  # 返回空列表以避免后续错误
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []  # 返回空列表



# def merge_txt_files(folder_name, base_dir):
#     '''
#     folder_name: 需要提取的文件夹名称
#     base_dir: 基础文件夹路径
#     '''
#     folder_path = os.path.join(base_dir, folder_name)
    
#     if not os.path.exists(folder_path):
#         print(f"Folder {folder_path} does not exist.")
#         return
    
#     # 遍历指定文件夹及其所有子文件夹
#     for root, dirs, files in os.walk(folder_path):
#         merged_data = []  # 用于存储当前子文件夹合并后的数据

#         for filename in files:
#             if filename.endswith('.txt'):
#                 file_path = os.path.join(root, filename)
#                 print(f'Processing file: {file_path}...')
                
#                 with open(file_path, 'r', encoding='utf-8') as fp:
#                     # 假设每个文件的内容是一个数组
#                     try:
#                         data = json.load(fp)  # 读取数组
#                         merged_data.extend(data)  # 合并到大数组中
#                     except json.JSONDecodeError:
#                         print(f"Error decoding JSON from {file_path}")

#         # 保存合并后的数据到当前子文件夹
#         if merged_data:  # 如果有合并的数据
#             merged_file_path = os.path.join(root, 'merged_data.json')
#             with open(merged_file_path, 'w', encoding='utf-8') as fp:
#                 json.dump(merged_data, fp, ensure_ascii=False, indent=4)
#             print(f'Merged data saved to: {merged_file_path}')




def merge_txt_files(folder_name, base_dir):
    '''
    folder_name: 需要提取的文件夹名称
    base_dir: 基础文件夹路径
    '''
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    # 遍历指定文件夹及其所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        merged_data = []  # 用于存储当前子文件夹合并后的数据
        subfolder_name = os.path.basename(root)  # 获取当前子文件夹的名字

        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                print(f'Processing file: {file_path}...')
                
                with open(file_path, 'r', encoding='utf-8') as fp:
                    try:
                        data = json.load(fp)  # 读取数组
                        # 为每个 JSON 对象添加 "file" 字段
                        for item in data:
                            item['file'] = subfolder_name  # 添加子文件夹名字
                        merged_data.extend(data)  # 合并到大数组中
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {file_path}")

        # 保存合并后的数据到当前子文件夹
        if merged_data:  # 如果有合并的数据
            merged_file_path = os.path.join(root, 'merged_data.json')
            with open(merged_file_path, 'w', encoding='utf-8') as fp:
                json.dump(merged_data, fp, ensure_ascii=False, indent=4)
            print(f'Merged data saved to: {merged_file_path}')





def extract_contents_from_folder(folder_name, base_dir):
    '''
    folder_name: 需要提取的文件夹名称
    base_dir: 基础文件夹路径
    '''
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return []
    
    print(f'Processing folder: {folder_name}...')

    merge_txt_files(folder_name, base_dir)

    # 遍历指定文件夹下的所有子文件夹
    for subfolder in tqdm(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        if os.path.isdir(subfolder_path):  # 确保是文件夹
#            final_block_path = os.path.join(subfolder_path, 'final_block.txt')
            merged_data_path = os.path.join(subfolder_path, 'merged_data.json')
            
            # if os.path.exists(final_block_path):
            #     final_block_list = read_final_block(final_block_path)
            #     if final_block_list:
            #         process_embeddings(final_block_list, folder_path ) 
            # else:
            #     print(f"{final_block_path} does not exist.")
            if os.path.exists(merged_data_path):
                with open(merged_data_path, 'r', encoding='utf-8') as fp:
                    try:
                        merged_data = json.load(fp)  # 读取 JSON 数据
                        if merged_data:
                            process_embeddings(merged_data, folder_path) 
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {merged_data_path}")
            else:
                print(f"{merged_data_path} does not exist.")

    return  # 直接返回，不需要返回合并的内容




client = OpenAI(
    api_key="sk-3796521366ac49daaf5041878d17eed0",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)




def process_embeddings(final_block_list, folder_path):
    '''处理提取的内容，生成向量嵌入并保存到指定文件夹'''
    results = []

    # 检查内容是否为空
    if not final_block_list:
        print("没有可处理的内容。")
        return

    # 如果文件夹不存在则创建
    os.makedirs(folder_path, exist_ok=True)

    # 遍历内容列表
    for obj in final_block_list: # 将 'object' 改为 'obj' 以避免与内置类型冲突
        # 检查内容是否为字典
        if isinstance(obj, dict):
            title = obj.get("title", "")
            text_content = "\n".join(obj.get("content", []))  # 将内容连接为字符串
            file_name = obj.get("file", "") 

            # 生成向量嵌入
            try:
                embedding = client.embeddings.create(
                    model="text-embedding-v4",
                    input=text_content,
                    dimensions=1024,
                    encoding_format="float"
                )

                # 从 Embedding 对象中提取实际的向量（浮点数列表）
                # embedding.data 是一个 Embedding 对象的列表，每个对象都有一个 'embedding' 属性
                if embedding.data and hasattr(embedding.data[0], 'embedding'):
                    vector = embedding.data[0].embedding
                else:
                    print(f"警告: 无法提取标题为 '{title}' 的嵌入向量。已跳过此项。")
                    continue

                results.append({
                    "title": title,
                    "file": file_name,
                    "text_content": text_content,
                    "embedding": vector  # 现在 'vector' 是一个浮点数列表，可以直接进行 JSON 序列化
                })
            except Exception as e:
                print(f"为标题 '{title}' 生成嵌入向量时出错: {e}。已跳过此代码块。")
                continue
        else:
            print(f"内容格式异常，预期为字典，实际为 {type(obj).__name__}。已跳过。")
            continue

    # 将结果保存到指定文件夹
    output_file_path = os.path.join(folder_path, 'embeddings.json')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"嵌入向量已保存到: {output_file_path}")



def extract_and_process(folder_names, base_dir):
    '''
    folder_names: 需要提取的文件夹名称列表
    base_dir: 基础文件夹路径
    '''
    for folder_name in folder_names:
        # 提取当前文件夹的内容
        all_contents = extract_contents_from_folder(folder_name, base_dir)
        




if __name__ == '__main__':
    folder_names = ['m_00008a']  # 需要提取的文件夹名称列表
    base_dir = '/data2/cwli16/2025FinancialConsistency/DATA/测试A集_clean/materials'  # 替换为实际路径
    
    extract_and_process(folder_names, base_dir)


