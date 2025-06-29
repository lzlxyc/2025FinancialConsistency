from langchain.retrievers import BM25Retriever
from langchain.schema import Document

import jieba


'''
实现了一个基于 BM25 (Best Matching 25) 算法的文档检索系统。
用于根据查询（query）和文档之间的匹配程度来计算文档的相关性分数。
'''
class BM25(object):

    def __init__(self, documents:list):
        '''
        构造函数初始化了一个 BM25 对象，接受一个文档列表 documents 作为输入。
        对于每个文档（以行的形式提供），先进行一些清理操作（去除多余的换行符和空格）。
        如果文档内容长度小于5个字符，则跳过该文档。
        使用 jieba.cut_for_search(line) 对文档进行中文分词，并将分词结果连接为一个空格分隔的字符串。
        分词后的文档存储为 Document 对象，并附加一个 metadata 字典，其中包含文档的唯一 id（通过 idx 索引获得）。
        对于原始文档（未经过分词），也将其保存到 full_docs 列表中，以便后续使用原始文档内容进行输出。
        self.documents 保存了分词后的文档集合，self.full_documents 保存了未处理的原始文档。
        最后，调用 _init_bm25 方法初始化 BM25 检索器。
        '''
        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if(len(line)<5):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            # docs.append(Document(page_content=tokens, metadata={"id": idx, "cate":words[1],"pageid":words[2]}))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            # full_docs.append(Document(page_content=words[0], metadata={"id": idx, "cate":words[1], "pageid":words[2]}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))

        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()


    # 初始化BM25的知识库
    def _init_bm25(self):
        '''
        这个方法初始化 BM25 检索器（BM25Retriever）并返回它。from_documents 方法可能是将输入的分词后文档列表（self.documents）
        转换为 BM25 检索器可以理解的格式。这使得 BM25 算法能够使用该集合进行检索操作。
        '''
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        '''
        该方法接受一个查询 query 和一个参数 topk，返回与查询最相关的前 topk 个文档。
        首先，设置 BM25 检索器的 k 参数为 topk，表示检索返回的文档数量。
        将查询 query 进行中文分词（与文档的分词方式一致），并将分词结果连接为一个空格分隔的字符串。
        调用 self.retriever.get_relevant_documents(query) 获取与查询相关的文档。
        get_relevant_documents 返回的是与查询最相关的文档索引列表（可能按相关性排序）。
        然后，通过这些索引，查找原始的文档 full_documents 并将其保存到 ans 列表中。
        最后，返回 ans 列表，其中包含了最相关的文档。
        '''
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []

        for line in ans_docs:
            print(line.metadata["id"])
            ans.append(line.page_content)
        return ans



if __name__ == "__main__":
    data = '''
    （一）故意行为或违法犯罪行为：投保人对被保险人的故意杀害或故意伤害；被保险人故意犯罪或抗拒依法采取的刑事强制措施、被政府依法拘禁或入狱期间；被保险人自杀、自伤，但被保险人自杀时为无民事行为能力人的除外；醉酒，服用、吸食或注射毒品；未遵医嘱，擅自服用、涂用、注射药物；被保险人酒后驾驶、无有效驾驶证驾驶（释义十二）或者驾驶无有效行驶证（释义十三）的机动交通工具；因被保险人挑衅或故意行为而导致的打斗、被袭击、被谋杀；
    （二）被保险人患性病，传染病（释义十四）、精神和行为障碍，遗传性疾病，先天性畸形、变形或染色体异常（依据世界卫生组织《疾病和有关健康问题的国际统计分类》（ICD-10）（释义十五）确定）；
    （三）康复治疗（释义十六）、休养或疗养、健康体检、隔离治疗、保健食品及用品；
    （四）康复治疗辅助装置或用具（包括义肢、轮椅、拐杖、助听器、眼镜或隐性眼镜、义眼、矫形支架等）及其安装、非处方医疗器械及其安装；
    （五）被保险人所患既往症（释义十七），及保险单中特别约定的除外疾病引起的相关费用，但投保时保险人已知晓并做出书面认可的除外；等待期内被保险人确诊的相关疾病；
    （六）被保险人感染艾滋病病毒或患艾滋病（释义十八），但在本合同有效期内，因职业原因导致人类免疫缺陷病毒（HIV）感染（释义十九）、输血原因导致人类免疫缺陷病毒（HIV）感染（释义二十）或器官移植原因导致人类免疫缺陷病毒（HIV）感染（释义二十一）不在此限；
    （七）战争、军事行动、暴乱或者武装叛乱；核爆炸、核辐射或者核污染、化学污染；
    （八）第三方服务商和医疗服务形式以外发生的看护费用；
    （九）不属于保险人指定或认可的第三方服务商发生的看护费用。'''

    bm25 = BM25(data.split('\n'))
    res = bm25.GetBM25TopK("取出与责任免除相关的信息", 2)
    print(res)