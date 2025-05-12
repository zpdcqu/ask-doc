#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# 导入JSON处理模块
import json
# 导入操作系统功能模块
import os
# 导入系统功能模块
import sys
# 导入时间处理模块
import time
# 导入命令行参数解析模块
import argparse
# 导入默认字典数据结构
from collections import defaultdict

# 导入LLM类型枚举
from api.db import LLMType
# 导入LLM服务包
from api.db.services.llm_service import LLMBundle
# 导入知识库服务
from api.db.services.knowledgebase_service import KnowledgebaseService
# 导入API设置
from api import settings
# 导入UUID生成工具
from api.utils import get_uuid
# 导入NLP处理函数
from rag.nlp import tokenize, search
# 导入评估函数
from ranx import evaluate
# 导入查询关联和运行结果类
from ranx import Qrels, Run
# 导入数据处理库
import pandas as pd
# 导入进度条显示库
from tqdm import tqdm

# 声明全局变量
global max_docs
# 初始化最大文档数为系统最大整数
max_docs = sys.maxsize


# 定义基准测试类
class Benchmark:
    # 初始化方法，接收知识库ID
    def __init__(self, kb_id):
        # 存储知识库ID
        self.kb_id = kb_id
        # 获取知识库对象
        e, self.kb = KnowledgebaseService.get_by_id(kb_id)
        # 获取相似度阈值
        self.similarity_threshold = self.kb.similarity_threshold
        # 获取向量相似度权重
        self.vector_similarity_weight = self.kb.vector_similarity_weight
        # 创建嵌入模型
        self.embd_mdl = LLMBundle(self.kb.tenant_id, LLMType.EMBEDDING, llm_name=self.kb.embd_id, lang=self.kb.language)
        # 初始化租户ID为空字符串
        self.tenant_id = ''
        # 初始化索引名为空字符串
        self.index_name = ''
        # 初始化索引状态为未初始化
        self.initialized_index = False

    # 获取检索结果的方法
    def _get_retrieval(self, qrels):
        # 等待ES和Infinity索引准备就绪
        time.sleep(20)
        # 创建默认字典存储运行结果
        run = defaultdict(dict)
        # 获取查询列表
        query_list = list(qrels.keys())
        # 遍历每个查询
        for query in query_list:
            # 使用检索器获取排名结果
            ranks = settings.retrievaler.retrieval(query, self.embd_mdl, self.tenant_id, [self.kb.id], 1, 30,
                                            0.0, self.vector_similarity_weight)
            # 如果没有检索到结果，删除该查询
            if len(ranks["chunks"]) == 0:
                print(f"deleted query: {query}")
                del qrels[query]
                continue
            # 处理检索结果
            for c in ranks["chunks"]:
                # 移除向量数据以节省空间
                c.pop("vector", None)
                # 存储相似度得分
                run[query][c["chunk_id"]] = c["similarity"]
        # 返回运行结果
        return run

    # 嵌入文档的方法
    def embedding(self, docs):
        # 提取文档内容
        texts = [d["content_with_weight"] for d in docs]
        # 使用嵌入模型编码文本
        embeddings, _ = self.embd_mdl.encode(texts)
        # 确保文档数量与嵌入数量一致
        assert len(docs) == len(embeddings)
        # 初始化向量大小
        vector_size = 0
        # 遍历文档和嵌入
        for i, d in enumerate(docs):
            # 获取当前文档的嵌入向量
            v = embeddings[i]
            # 更新向量大小
            vector_size = len(v)
            # 将向量添加到文档中
            d["q_%d_vec" % len(v)] = v
        # 返回处理后的文档和向量大小
        return docs, vector_size

    # 初始化索引的方法
    def init_index(self, vector_size: int):
        # 如果索引已初始化，直接返回
        if self.initialized_index:
            return
        # 检查索引是否存在，如存在则删除
        if settings.docStoreConn.indexExist(self.index_name, self.kb_id):
            settings.docStoreConn.deleteIdx(self.index_name, self.kb_id)
        # 创建新索引
        settings.docStoreConn.createIdx(self.index_name, self.kb_id, vector_size)
        # 标记索引已初始化
        self.initialized_index = True

    # 为MS MARCO数据集创建索引的方法
    def ms_marco_index(self, file_path, index_name):
        # 创建默认字典存储查询关联
        qrels = defaultdict(dict)
        # 创建默认字典存储文本
        texts = defaultdict(dict)
        # 初始化文档计数
        docs_count = 0
        # 初始化文档列表
        docs = []
        # 获取并排序文件列表
        filelist = sorted(os.listdir(file_path))

        # 遍历文件列表
        for fn in filelist:
            # 如果达到最大文档数，跳出循环
            if docs_count >= max_docs:
                break
            # 只处理parquet文件
            if not fn.endswith(".parquet"):
                continue
            # 读取parquet文件
            data = pd.read_parquet(os.path.join(file_path, fn))
            # 遍历数据行
            for i in tqdm(range(len(data)), colour="green", desc="Tokenizing:" + fn):
                # 如果达到最大文档数，跳出循环
                if docs_count >= max_docs:
                    break
                # 获取查询文本
                query = data.iloc[i]['query']
                # 遍历段落及其相关性
                for rel, text in zip(data.iloc[i]['passages']['is_selected'], data.iloc[i]['passages']['passage_text']):
                    # 创建文档字典
                    d = {
                        "id": get_uuid(),
                        "kb_id": self.kb.id,
                        "docnm_kwd": "xxxxx",
                        "doc_id": "ksksks"
                    }
                    # 对文本进行分词处理
                    tokenize(d, text, "english")
                    # 添加到文档列表
                    docs.append(d)
                    # 存储文本
                    texts[d["id"]] = text
                    # 存储查询关联
                    qrels[query][d["id"]] = int(rel)
                # 当文档数量达到32时，进行批处理
                if len(docs) >= 32:
                    # 增加文档计数
                    docs_count += len(docs)
                    # 对文档进行嵌入处理
                    docs, vector_size = self.embedding(docs)
                    # 初始化索引
                    self.init_index(vector_size)
                    # 插入文档到索引
                    settings.docStoreConn.insert(docs, self.index_name, self.kb_id)
                    # 重置文档列表
                    docs = []

        # 处理剩余文档
        if docs:
            # 对剩余文档进行嵌入处理
            docs, vector_size = self.embedding(docs)
            # 初始化索引
            self.init_index(vector_size)
            # 插入文档到索引
            settings.docStoreConn.insert(docs, self.index_name, self.kb_id)
        # 返回查询关联和文本
        return qrels, texts

    # 为TriviaQA数据集创建索引的方法
    def trivia_qa_index(self, file_path, index_name):
        # 创建默认字典存储查询关联
        qrels = defaultdict(dict)
        # 创建默认字典存储文本
        texts = defaultdict(dict)
        # 初始化文档计数
        docs_count = 0
        # 初始化文档列表
        docs = []
        # 获取并排序文件列表
        filelist = sorted(os.listdir(file_path))
        # 遍历文件列表
        for fn in filelist:
            # 如果达到最大文档数，跳出循环
            if docs_count >= max_docs:
                break
            # 只处理parquet文件
            if not fn.endswith(".parquet"):
                continue
            # 读取parquet文件
            data = pd.read_parquet(os.path.join(file_path, fn))
            # 遍历数据行
            for i in tqdm(range(len(data)), colour="green", desc="Indexing:" + fn):
                # 如果达到最大文档数，跳出循环
                if docs_count >= max_docs:
                    break
                # 获取问题文本
                query = data.iloc[i]['question']
                # 遍历搜索结果及其排名
                for rel, text in zip(data.iloc[i]["search_results"]['rank'],
                                     data.iloc[i]["search_results"]['search_context']):
                    # 创建文档字典
                    d = {
                        "id": get_uuid(),
                        "kb_id": self.kb.id,
                        "docnm_kwd": "xxxxx",
                        "doc_id": "ksksks"
                    }
                    # 对文本进行分词处理
                    tokenize(d, text, "english")
                    # 添加到文档列表
                    docs.append(d)
                    # 存储文本
                    texts[d["id"]] = text
                    # 存储查询关联
                    qrels[query][d["id"]] = int(rel)
                # 当文档数量达到32时，进行批处理
                if len(docs) >= 32:
                    # 增加文档计数
                    docs_count += len(docs)
                    # 对文档进行嵌入处理
                    docs, vector_size = self.embedding(docs)
                    # 初始化索引
                    self.init_index(vector_size)
                    # 插入文档到索引
                    settings.docStoreConn.insert(docs,self.index_name)
                    # 重置文档列表
                    docs = []

        # 对剩余文档进行嵌入处理
        docs, vector_size = self.embedding(docs)
        # 初始化索引
        self.init_index(vector_size)
        # 插入文档到索引
        settings.docStoreConn.insert(docs, self.index_name)
        # 返回查询关联和文本
        return qrels, texts

    # 为MIRACL数据集创建索引的方法
    def miracl_index(self, file_path, corpus_path, index_name):
        # 初始化语料库总字典
        corpus_total = {}
        # 遍历语料库文件
        for corpus_file in os.listdir(corpus_path):
            # 读取JSON行文件
            tmp_data = pd.read_json(os.path.join(corpus_path, corpus_file), lines=True)
            # 遍历数据行
            for index, i in tmp_data.iterrows():
                # 存储文档ID和文本
                corpus_total[i['docid']] = i['text']

        # 初始化主题总字典
        topics_total = {}
        # 遍历主题文件
        for topics_file in os.listdir(os.path.join(file_path, 'topics')):
            # 跳过测试文件
            if 'test' in topics_file:
                continue
            # 读取CSV文件
            tmp_data = pd.read_csv(os.path.join(file_path, 'topics', topics_file), sep='\t', names=['qid', 'query'])
            # 遍历数据行
            for index, i in tmp_data.iterrows():
                # 存储查询ID和查询文本
                topics_total[i['qid']] = i['query']

        # 创建默认字典存储查询关联
        qrels = defaultdict(dict)
        # 创建默认字典存储文本
        texts = defaultdict(dict)
        # 初始化文档计数
        docs_count = 0
        # 初始化文档列表
        docs = []
        # 遍历查询关联文件
        for qrels_file in os.listdir(os.path.join(file_path, 'qrels')):
            # 跳过测试文件
            if 'test' in qrels_file:
                continue
            # 如果达到最大文档数，跳出循环
            if docs_count >= max_docs:
                break

            # 读取CSV文件
            tmp_data = pd.read_csv(os.path.join(file_path, 'qrels', qrels_file), sep='\t',
                                   names=['qid', 'Q0', 'docid', 'relevance'])
            # 遍历数据行
            for i in tqdm(range(len(tmp_data)), colour="green", desc="Indexing:" + qrels_file):
                # 如果达到最大文档数，跳出循环
                if docs_count >= max_docs:
                    break
                # 获取查询文本
                query = topics_total[tmp_data.iloc[i]['qid']]
                # 获取文档文本
                text = corpus_total[tmp_data.iloc[i]['docid']]
                # 获取相关性得分
                rel = tmp_data.iloc[i]['relevance']
                # 创建文档字典
                d = {
                    "id": get_uuid(),
                    "kb_id": self.kb.id,
                    "docnm_kwd": "xxxxx",
                    "doc_id": "ksksks"
                }
                # 对文本进行分词处理
                tokenize(d, text, 'english')
                # 添加到文档列表
                docs.append(d)
                # 存储文本
                texts[d["id"]] = text
                # 存储查询关联
                qrels[query][d["id"]] = int(rel)
                # 当文档数量达到32时，进行批处理
                if len(docs) >= 32:
                    # 增加文档计数
                    docs_count += len(docs)
                    # 对文档进行嵌入处理
                    docs, vector_size = self.embedding(docs)
                    # 初始化索引
                    self.init_index(vector_size)
                    # 插入文档到索引
                    settings.docStoreConn.insert(docs, self.index_name)
                    # 重置文档列表
                    docs = []

        # 对剩余文档进行嵌入处理
        docs, vector_size = self.embedding(docs)
        # 初始化索引
        self.init_index(vector_size)
        # 插入文档到索引
        settings.docStoreConn.insert(docs, self.index_name)
        # 返回查询关联和文本
        return qrels, texts

    # 保存结果的方法
    def save_results(self, qrels, run, texts, dataset, file_path):
        # 初始化结果列表
        keep_result = []
        # 获取运行结果的键列表
        run_keys = list(run.keys())
        # 遍历运行结果键
        for run_i in tqdm(range(len(run_keys)), desc="Calculating ndcg@10 for single query"):
            # 获取当前键
            key = run_keys[run_i]
            # 计算并存储结果
            keep_result.append({'query': key, 'qrel': qrels[key], 'run': run[key],
                                'ndcg@10': evaluate({key: qrels[key]}, {key: run[key]}, "ndcg@10")})
        # 按NDCG@10得分排序结果
        keep_result = sorted(keep_result, key=lambda kk: kk['ndcg@10'])
        # 打开文件写入结果
        with open(os.path.join(file_path, dataset + 'result.md'), 'w', encoding='utf-8') as f:
            # 写入标题
            f.write('## Score For Every Query\n')
            # 遍历结果
            for keep_result_i in keep_result:
                # 写入查询和得分
                f.write('### query: ' + keep_result_i['query'] + ' ndcg@10:' + str(keep_result_i['ndcg@10']) + '\n')
                # 提取得分并排序
                scores = [[i[0], i[1]] for i in keep_result_i['run'].items()]
                scores = sorted(scores, key=lambda kk: kk[1])
                # 写入前10个结果
                for score in scores[:10]:
                    f.write('- text: ' + str(texts[score[0]]) + '\t qrel: ' + str(score[1]) + '\n')
        # 保存查询关联为JSON
        json.dump(qrels, open(os.path.join(file_path, dataset + '.qrels.json'), "w+", encoding='utf-8'), indent=2)
        # 保存运行结果为JSON
        json.dump(run, open(os.path.join(file_path, dataset + '.run.json'), "w+", encoding='utf-8'), indent=2)
        # 打印保存路径
        print(os.path.join(file_path, dataset + '_result.md'), 'Saved!')

    # 调用方法，执行基准测试
    def __call__(self, dataset, file_path, miracl_corpus=''):
        # 处理MS MARCO数据集
        if dataset == "ms_marco_v1.1":
            # 设置租户ID
            self.tenant_id = "benchmark_ms_marco_v11"
            # 获取索引名
            self.index_name = search.index_name(self.tenant_id)
            # 创建索引并获取查询关联和文本
            qrels, texts = self.ms_marco_index(file_path, "benchmark_ms_marco_v1.1")
            # 获取检索结果
            run = self._get_retrieval(qrels)
            # 打印评估结果
            print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
            # 保存结果
            self.save_results(qrels, run, texts, dataset, file_path)
        # 处理TriviaQA数据集
        if dataset == "trivia_qa":
            # 设置租户ID
            self.tenant_id = "benchmark_trivia_qa"
            # 获取索引名
            self.index_name = search.index_name(self.tenant_id)
            # 创建索引并获取查询关联和文本
            qrels, texts = self.trivia_qa_index(file_path, "benchmark_trivia_qa")
            # 获取检索结果
            run = self._get_retrieval(qrels)
            # 打印评估结果
            print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
            # 保存结果
            self.save_results(qrels, run, texts, dataset, file_path)
        # 处理MIRACL数据集
        if dataset == "miracl":
            # 遍历支持的语言
            for lang in ['ar', 'bn', 'de', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th',
                         'yo', 'zh']:
                # 检查数据集目录是否存在
                if not os.path.isdir(os.path.join(file_path, 'miracl-v1.0-' + lang)):
                    print('Directory: ' + os.path.join(file_path, 'miracl-v1.0-' + lang) + ' not found!')
                    continue
                # 检查查询关联目录是否存在
                if not os.path.isdir(os.path.join(file_path, 'miracl-v1.0-' + lang, 'qrels')):
                    print('Directory: ' + os.path.join(file_path, 'miracl-v1.0-' + lang, 'qrels') + 'not found!')
                    continue
                # 检查主题目录是否存在
                if not os.path.isdir(os.path.join(file_path, 'miracl-v1.0-' + lang, 'topics')):
                    print('Directory: ' + os.path.join(file_path, 'miracl-v1.0-' + lang, 'topics') + 'not found!')
                    continue
                # 检查语料库目录是否存在
                if not os.path.isdir(os.path.join(miracl_corpus, 'miracl-corpus-v1.0-' + lang)):
                    print('Directory: ' + os.path.join(miracl_corpus, 'miracl-corpus-v1.0-' + lang) + ' not found!')
                    continue
                # 设置租户ID
                self.tenant_id = "benchmark_miracl_" + lang
                # 获取索引名
                self.index_name = search.index_name(self.tenant_id)
                # 重置索引状态
                self.initialized_index = False
                # 创建索引并获取查询关联和文本
                qrels, texts = self.miracl_index(os.path.join(file_path, 'miracl-v1.0-' + lang),
                                                 os.path.join(miracl_corpus, 'miracl-corpus-v1.0-' + lang),
                                                 "benchmark_miracl_" + lang)
                # 获取检索结果
                run = self._get_retrieval(qrels)
                # 打印评估结果
                print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
                # 保存结果
                self.save_results(qrels, run, texts, dataset, file_path)


# 主程序入口
if __name__ == '__main__':
    # 打印标题
    print('*****************RAGFlow Benchmark*****************')
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(usage="benchmark.py <max_docs> <kb_id> <dataset> <dataset_path> [<miracl_corpus_path>])", description='RAGFlow Benchmark')
    # 添加最大文档数参数
    parser.add_argument('max_docs', metavar='max_docs', type=int, help='max docs to evaluate')
    # 添加知识库ID参数
    parser.add_argument('kb_id', metavar='kb_id', help='knowledgebase id')
    # 添加数据集名称参数
    parser.add_argument('dataset', metavar='dataset', help='dataset name, shall be one of ms_marco_v1.1(https://huggingface.co/datasets/microsoft/ms_marco), trivia_qa(https://huggingface.co/datasets/mandarjoshi/trivia_qa>), miracl(https://huggingface.co/datasets/miracl/miracl')
    # 添加数据集路径参数
    parser.add_argument('dataset_path', metavar='dataset_path', help='dataset path')
    # 添加MIRACL语料库路径参数（可选）
    parser.add_argument('miracl_corpus_path', metavar='miracl_corpus_path', nargs='?', default="", help='miracl corpus path. Only needed when dataset is miracl')

    # 解析命令行参数
    args = parser.parse_args()
    # 设置最大文档数
    max_docs = args.max_docs
    # 获取知识库ID
    kb_id = args.kb_id
    # 创建基准测试实例
    ex = Benchmark(kb_id)

    # 获取数据集名称
    dataset = args.dataset
    # 获取数据集路径
    dataset_path = args.dataset_path

    # 处理MS MARCO或TriviaQA数据集
    if dataset == "ms_marco_v1.1" or dataset == "trivia_qa":
        ex(dataset, dataset_path)
    # 处理MIRACL数据集
    elif dataset == "miracl":
        # 检查参数数量
        if len(args) < 5:
            print('Please input the correct parameters!')
            exit(1)
        # 获取MIRACL语料库路径
        miracl_corpus_path = args[4]
        # 执行基准测试
        ex(dataset, dataset_path, miracl_corpus=args.miracl_corpus_path)
    # 处理不支持的数据集
    else:
        print("Dataset: ", dataset, "not supported!")
