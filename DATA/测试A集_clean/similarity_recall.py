import pandas as pd
import re
import markdown
import glob, os, re
from tqdm import tqdm
import pandas as pd
import json
import numpy as np


with open('/data2/cwli16/2025FinancialConsistency/DATA/测试A集_clean/materials/m_00008a/embeddings.json', 'r', encoding='utf-8') as f:
    embeddings_data = json.load(f)

def cosine_similarity(vec_a, vec_b):
    """计算余弦相似度"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

#****************没有限制不能比对来自同一文件夹里面的条款**********************************
# def find_similar_items(embeddings_data, file_path, threshold=0.7):
#     similar_items = []
    
#     # 遍历每对条款
#     for i in range(len(embeddings_data)):
#         for j in range(i + 1, len(embeddings_data)):  # 只比较一次
#             title_a = embeddings_data[i]["title"]
#             text_content_a = embeddings_data[i]["text_content"]
#             embedding_a = embeddings_data[i]["embedding"]
            
#             title_b = embeddings_data[j]["title"]
#             text_content_b = embeddings_data[j]["text_content"]
#             embedding_b = embeddings_data[j]["embedding"]
            
#             similarity = cosine_similarity(embedding_a, embedding_b)
            
#             if similarity > threshold:
#                 similar_items.append((title_a, text_content_a, title_b, text_content_b, similarity))
   
#     if os.path.isdir(file_path):
#         file_path = os.path.join(file_path, 'similar_items.txt')  # 默认文件名

#     # 打印和保存结果
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for title_a, text_content_a, title_b, text_content_b, similarity in similar_items:
#             print(f"条款 A 标题: {title_a}, 相似度: {similarity:.4f}")
#             print(f"条款 A 原文内容: {text_content_a}\n")
#             print(f"条款 B 标题: {title_b}, 相似度: {similarity:.4f}")
#             print(f"条款 B 原文内容: {text_content_b}\n")
            
#             f.write(f"条款 A 标题: {title_a}, 相似度: {similarity:.4f}\n")
#             f.write(f"条款 A 原文内容: {text_content_a}\n\n")
#             f.write(f"条款 B 标题: {title_b}, 相似度: {similarity:.4f}\n")
#             f.write(f"条款 B 原文内容: {text_content_b}\n\n")
#             f.write("-" * 40 + "\n")  # 分隔线





#******************限制了不能比对来自同一个文件夹里的条款**********************************


def find_similar_items(embeddings_data, file_path, threshold=0.7):
    similar_items = []

    # 遍历每对条款
    for i in range(len(embeddings_data)):
        for j in range(i + 1, len(embeddings_data)):  # 只比较一次
            title_a = embeddings_data[i]["title"]
            text_content_a = embeddings_data[i]["text_content"]
            embedding_a = embeddings_data[i]["embedding"]
            file_a = embeddings_data[i]["file"]  # 获取 file 字段

            title_b = embeddings_data[j]["title"]
            text_content_b = embeddings_data[j]["text_content"]
            embedding_b = embeddings_data[j]["embedding"]
            file_b = embeddings_data[j]["file"]  # 获取 file 字段

            # 检查 file 字段是否相同
            if file_a == file_b:
                continue  # 跳过同一文件夹的条款

            similarity = cosine_similarity(embedding_a, embedding_b)

            if similarity > threshold:
                similar_items.append((title_a, text_content_a, title_b, text_content_b, similarity))

    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, 'similar_items.txt')  # 默认文件名

    # 打印和保存结果
    with open(file_path, 'w', encoding='utf-8') as f:
        for title_a, text_content_a, title_b, text_content_b, similarity in similar_items:
            print(f"条款 A 标题: {title_a}, 相似度: {similarity:.4f}")
            print(f"条款 A 原文内容: {text_content_a}\n")
            print(f"条款 B 标题: {title_b}, 相似度: {similarity:.4f}")
            print(f"条款 B 原文内容: {text_content_b}\n")

            f.write(f"条款 A 标题: {title_a}, 相似度: {similarity:.4f}\n")
            f.write(f"条款 A 原文内容: {text_content_a}\n\n")
            f.write(f"条款 B 标题: {title_b}, 相似度: {similarity:.4f}\n")
            f.write(f"条款 B 原文内容: {text_content_b}\n\n")
            f.write("-" * 40 + "\n")  # 分隔线


output_file_path='/data2/cwli16/2025FinancialConsistency/DATA'
find_similar_items(embeddings_data, output_file_path)