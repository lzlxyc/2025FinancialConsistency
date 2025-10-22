import csv
import json

csv_path = "data/raw_data_b/['赔付 & 领取规则', '责任免除', '投保条款', '保障相关时间', '保障责任', '基础产品销售信息', '退保条款', '附加条款', '续保条款', '出险条款', '术语解释'].csv"
jsonl_path = "['赔付 & 领取规则', '责任免除', '投保条款', '保障相关时间', '保障责任', '基础产品销售信息', '退保条款', '附加条款', '续保条款', '出险条款', '术语解释'].jsonl"

with open(csv_path, encoding="utf-8") as f_csv, open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
    reader = csv.DictReader(f_csv)
    for row in reader:
        # ypred字符串转为布尔值
        result = row["ypred"].strip().lower() == "true"
        obj = {
            "material_id": row["material_id"],
            "rule_id": row["rule_id"],
            "result": result
        }
        f_jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")