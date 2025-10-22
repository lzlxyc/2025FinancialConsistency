import os

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

# 整理所有要删除的关键词（key + value）
delete_phrases = list(rule_clauses.keys()) + list(rule_clauses.values())

# 递归遍历所有 txt 文件
for root, dirs, files in os.walk("data/raw_data_b/materials"):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 删除所有 key 和 value 出现的文字
            for phrase in delete_phrases:
                content = content.replace(phrase, "")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)