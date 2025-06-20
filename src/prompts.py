PT_Extract_RuleInfo = f"提取出下面文本的{rule}条款（约定该产品不承担保险责任的情形）信息，要完整的、不要有遗漏的信息，更不要修改数据内容。如果相关的本文本不存在，则输出一个空字符串。"

- 基础产品销售信息：该保险产品的基础配置信息，包括产品名、附加的条款信息、销售限制等；
- 投保条款：投保过程中的缴费约定、投被保人条件限制等；
- 保障责任：约定该产品的保险责任细节，如保障范围、保险金额、增值服务等；
- 保障相关时间：约定该产品的各类时间信息，包括但不限于犹豫期、等待期、宽限期等；
- 赔付 & 领取规则：约定该产品的保险责任的赔付、给付、领取及免赔细节，如赔付年龄/比例/次数等；
- 责任免除：约定该产品不承担保险责任的情形；
- 续保条款：约定续保相关信息，包括但不限于续保条件、保证续保等；
- 退保条款：约定退保相关信息，包括但不限于退保条件、退保手续费等；
- 出险条款：约定出险相关信息，包括但不限于出险地点、出险方式等；
- 附加条款：约定该产品的附加条款，如特别约定等；
- 术语解释：约定该产品的术语解释，如名词定义等；