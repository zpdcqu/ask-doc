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
# 导入json模块
import json
# 导入日志模块
import logging
# 导入defaultdict数据结构
from collections import defaultdict
# 导入深拷贝函数
from copy import deepcopy
# 导入json修复模块
import json_repair
# 导入pandas数据分析库
import pandas as pd

# 导入UUID生成工具
from api.utils import get_uuid
# 导入查询分析提示模板
from graphrag.query_analyze_prompt import PROMPTS
# 导入实体类型、LLM缓存和关系获取工具
from graphrag.utils import get_entity_type2sampels, get_llm_cache, set_llm_cache, get_relation
# 导入计算文本token数的工具
from rag.utils import num_tokens_from_string
# 导入排序表达式类
from rag.utils.doc_store_conn import OrderByExpr

# 导入搜索处理器和索引名称生成函数
from rag.nlp.search import Dealer, index_name


# 定义知识图谱搜索类，继承自Dealer
class KGSearch(Dealer):
    # 与LLM聊天的内部方法
    def _chat(self, llm_bdl, system, history, gen_conf):
        # 尝试从缓存获取响应
        response = get_llm_cache(llm_bdl.llm_name, system, history, gen_conf)
        # 如果缓存中有响应，直接返回
        if response:
            return response
        # 否则调用LLM进行聊天
        response = llm_bdl.chat(system, history, gen_conf)
        # 如果响应中包含错误标记，抛出异常
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        # 将响应存入缓存
        set_llm_cache(llm_bdl.llm_name, system, response, history, gen_conf)
        # 返回响应
        return response

    # 查询重写方法
    def query_rewrite(self, llm, question, idxnms, kb_ids):
        # 获取实体类型到样本的映射
        ty2ents = get_entity_type2sampels(idxnms, kb_ids)
        # 格式化提示模板
        hint_prompt = PROMPTS["minirag_query2kwd"].format(query=question,
                                                          TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2))
        # 调用LLM进行聊天
        result = self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {"temperature": .5})
        try:
            # 尝试解析JSON响应
            keywords_data = json_repair.loads(result)
            # 获取类型关键词
            type_keywords = keywords_data.get("answer_type_keywords", [])
            # 获取查询中的实体，最多5个
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]
            # 返回类型关键词和实体
            return type_keywords, entities_from_query
        except json_repair.JSONDecodeError:
            try:
                # 如果解析失败，尝试清理响应并重新解析
                result = result.replace(hint_prompt[:-1], '').replace('user', '').replace('model', '').strip()
                result = '{' + result.split('{')[1].split('}')[0] + '}'
                # 解析清理后的JSON
                keywords_data = json_repair.loads(result)
                # 获取类型关键词
                type_keywords = keywords_data.get("answer_type_keywords", [])
                # 获取查询中的实体，最多5个
                entities_from_query = keywords_data.get("entities_from_query", [])[:5]
                # 返回类型关键词和实体
                return type_keywords, entities_from_query
            # 处理解析错误
            except Exception as e:
                # 记录异常
                logging.exception(f"JSON parsing error: {result} -> {e}")
                # 抛出异常
                raise e

    # 从搜索结果中提取实体信息
    def _ent_info_from_(self, es_res, sim_thr=0.3):
        # 初始化结果字典
        res = {}
        # 定义需要获取的字段
        flds = ["content_with_weight", "_score", "entity_kwd", "rank_flt", "n_hop_with_weight"]
        # 从搜索结果中获取字段
        es_res = self.dataStore.getFields(es_res, flds)
        # 遍历搜索结果
        for _, ent in es_res.items():
            # 删除值为None的字段
            for f in flds:
                if f in ent and ent[f] is None:
                    del ent[f]
            # 如果相似度低于阈值，跳过
            if float(ent.get("_score", 0)) < sim_thr:
                continue
            # 如果实体关键词是列表，取第一个元素
            if isinstance(ent["entity_kwd"], list):
                ent["entity_kwd"] = ent["entity_kwd"][0]
            # 构建实体信息
            res[ent["entity_kwd"]] = {
                "sim": float(ent.get("_score", 0)),
                "pagerank": float(ent.get("rank_flt", 0)),
                "n_hop_ents": json.loads(ent.get("n_hop_with_weight", "[]")),
                "description": ent.get("content_with_weight", "{}")
            }
        # 返回结果
        return res

    # 从搜索结果中提取关系信息
    def _relation_info_from_(self, es_res, sim_thr=0.3):
        # 初始化结果字典
        res = {}
        # 从搜索结果中获取字段
        es_res = self.dataStore.getFields(es_res, ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd",
                                                   "weight_int"])
        # 遍历搜索结果
        for _, ent in es_res.items():
            # 如果相似度低于阈值，跳过
            if float(ent["_score"]) < sim_thr:
                continue
            # 对实体对进行排序
            f, t = sorted([ent["from_entity_kwd"], ent["to_entity_kwd"]])
            # 如果实体是列表，取第一个元素
            if isinstance(f, list):
                f = f[0]
            if isinstance(t, list):
                t = t[0]
            # 构建关系信息
            res[(f, t)] = {
                "sim": float(ent["_score"]),
                "pagerank": float(ent.get("weight_int", 0)),
                "description": ent["content_with_weight"]
            }
        # 返回结果
        return res

    # 通过关键词获取相关实体
    def get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        # 如果没有关键词，返回空字典
        if not keywords:
            return {}
        # 深拷贝过滤条件
        filters = deepcopy(filters)
        # 设置知识图谱关键词为实体
        filters["knowledge_graph_kwd"] = "entity"
        # 获取向量表示
        matchDense = self.get_vector(", ".join(keywords), emb_mdl, 1024, sim_thr)
        # 执行搜索
        es_res = self.dataStore.search(["content_with_weight", "entity_kwd", "rank_flt"], [], filters, [matchDense],
                                       OrderByExpr(), 0, N,
                                       idxnms, kb_ids)
        # 返回实体信息
        return self._ent_info_from_(es_res, sim_thr)

    # 通过文本获取相关关系
    def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        # 如果没有文本，返回空字典
        if not txt:
            return {}
        # 深拷贝过滤条件
        filters = deepcopy(filters)
        # 设置知识图谱关键词为关系
        filters["knowledge_graph_kwd"] = "relation"
        # 获取向量表示
        matchDense = self.get_vector(txt, emb_mdl, 1024, sim_thr)
        # 执行搜索
        es_res = self.dataStore.search(
            ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd", "weight_int"],
            [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids)
        # 返回关系信息
        return self._relation_info_from_(es_res, sim_thr)

    # 通过类型获取相关实体
    def get_relevant_ents_by_types(self, types, filters, idxnms, kb_ids, N=56):
        # 如果没有类型，返回空字典
        if not types:
            return {}
        # 深拷贝过滤条件
        filters = deepcopy(filters)
        # 设置知识图谱关键词为实体
        filters["knowledge_graph_kwd"] = "entity"
        # 设置实体类型关键词
        filters["entity_type_kwd"] = types
        # 创建排序表达式
        ordr = OrderByExpr()
        # 按PageRank降序排序
        ordr.desc("rank_flt")
        # 执行搜索
        es_res = self.dataStore.search(["entity_kwd", "rank_flt"], [], filters, [], ordr, 0, N,
                                       idxnms, kb_ids)
        # 返回实体信息
        return self._ent_info_from_(es_res, 0)

    # 检索方法
    def retrieval(self, question: str,
               tenant_ids: str | list[str],
               kb_ids: list[str],
               emb_mdl,
               llm,
               max_token: int = 8196,
               ent_topn: int = 6,
               rel_topn: int = 6,
               comm_topn: int = 1,
               ent_sim_threshold: float = 0.3,
               rel_sim_threshold: float = 0.3,
               ):
        # 保存问题
        qst = question
        # 获取过滤条件
        filters = self.get_filters({"kb_ids": kb_ids})
        # 如果租户ID是字符串，转换为列表
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
        # 生成索引名称列表
        idxnms = [index_name(tid) for tid in tenant_ids]
        # 初始化类型关键词
        ty_kwds = []
        try:
            # 尝试重写查询
            ty_kwds, ents = self.query_rewrite(llm, qst, [index_name(tid) for tid in tenant_ids], kb_ids)
            # 记录日志
            logging.info(f"Q: {qst}, Types: {ty_kwds}, Entities: {ents}")
        except Exception as e:
            # 记录异常
            logging.exception(e)
            # 如果查询重写失败，使用原问题作为实体
            ents = [qst]
            pass

        # 通过关键词获取相关实体
        ents_from_query = self.get_relevant_ents_by_keywords(ents, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold)
        # 通过类型获取相关实体
        ents_from_types = self.get_relevant_ents_by_types(ty_kwds, filters, idxnms, kb_ids, 10000)
        # 通过文本获取相关关系
        rels_from_txt = self.get_relevant_relations_by_txt(qst, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold)
        # 初始化N跳路径字典
        nhop_pathes = defaultdict(dict)
        # 遍历查询得到的实体
        for _, ent in ents_from_query.items():
            # 获取N跳实体
            nhops = ent.get("n_hop_ents", [])
            # 如果N跳实体不是列表，记录警告并跳过
            if not isinstance(nhops, list):
                logging.warning(f"Abnormal n_hop_ents: {nhops}")
                continue
            # 遍历N跳实体
            for nbr in nhops:
                # 获取路径
                path = nbr["path"]
                # 获取权重
                wts = nbr["weights"]
                # 遍历路径中的每对实体
                for i in range(len(path) - 1):
                    # 获取起点和终点
                    f, t = path[i], path[i + 1]
                    # 如果路径已存在，累加相似度
                    if (f, t) in nhop_pathes:
                        nhop_pathes[(f, t)]["sim"] += ent["sim"] / (2 + i)
                    # 否则，初始化相似度
                    else:
                        nhop_pathes[(f, t)]["sim"] = ent["sim"] / (2 + i)
                    # 设置PageRank
                    nhop_pathes[(f, t)]["pagerank"] = wts[i]

        # 记录检索到的实体
        logging.info("Retrieved entities: {}".format(list(ents_from_query.keys())))
        # 记录检索到的关系
        logging.info("Retrieved relations: {}".format(list(rels_from_txt.keys())))
        # 记录从类型检索到的实体
        logging.info("Retrieved entities from types({}): {}".format(ty_kwds, list(ents_from_types.keys())))
        # 记录检索到的N跳路径
        logging.info("Retrieved N-hops: {}".format(list(nhop_pathes.keys())))

        # P(E|Q) => P(E) * P(Q|E) => pagerank * sim
        # 如果实体同时在类型检索结果中，增加相似度
        for ent in ents_from_types.keys():
            if ent not in ents_from_query:
                continue
            ents_from_query[ent]["sim"] *= 2

        # 遍历文本检索到的关系
        for (f, t) in rels_from_txt.keys():
            # 对实体对进行排序
            pair = tuple(sorted([f, t]))
            # 初始化分数
            s = 0
            # 如果关系在N跳路径中，增加相似度并删除N跳路径中的记录
            if pair in nhop_pathes:
                s += nhop_pathes[pair]["sim"]
                del nhop_pathes[pair]
            # 如果起点在类型检索结果中，增加分数
            if f in ents_from_types:
                s += 1
            # 如果终点在类型检索结果中，增加分数
            if t in ents_from_types:
                s += 1
            # 调整相似度
            rels_from_txt[(f, t)]["sim"] *= s + 1

        # 处理N跳路径中的关系（这些关系不是通过查询搜索得到的）
        for (f, t) in nhop_pathes.keys():
            # 初始化分数
            s = 0
            # 如果起点在类型检索结果中，增加分数
            if f in ents_from_types:
                s += 1
            # 如果终点在类型检索结果中，增加分数
            if t in ents_from_types:
                s += 1
            # 添加关系信息
            rels_from_txt[(f, t)] = {
                "sim": nhop_pathes[(f, t)]["sim"] * (s + 1),
                "pagerank": nhop_pathes[(f, t)]["pagerank"]
            }

        # 对实体按相似度*PageRank降序排序，并取前ent_topn个
        ents_from_query = sorted(ents_from_query.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[
                          :ent_topn]
        # 对关系按相似度*PageRank降序排序，并取前rel_topn个
        rels_from_txt = sorted(rels_from_txt.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[
                        :rel_topn]

        # 初始化实体和关系列表
        ents = []
        relas = []
        # 遍历排序后的实体
        for n, ent in ents_from_query:
            # 添加实体信息
            ents.append({
                "Entity": n,
                "Score": "%.2f" % (ent["sim"] * ent["pagerank"]),
                "Description": json.loads(ent["description"]).get("description", "") if ent["description"] else ""
            })
            # 减少可用token数
            max_token -= num_tokens_from_string(str(ents[-1]))
            # 如果token数不足，删除最后一个实体并退出循环
            if max_token <= 0:
                ents = ents[:-1]
                break

        # 遍历排序后的关系
        for (f, t), rel in rels_from_txt:
            # 如果关系没有描述，尝试获取
            if not rel.get("description"):
                for tid in tenant_ids:
                    rela = get_relation(tid, kb_ids, f, t)
                    if rela:
                        break
                else:
                    continue
                rel["description"] = rela["description"]
            # 获取描述
            desc = rel["description"]
            try:
                # 尝试解析JSON描述
                desc = json.loads(desc).get("description", "")
            except Exception:
                pass
            # 添加关系信息
            relas.append({
                "From Entity": f,
                "To Entity": t,
                "Score": "%.2f" % (rel["sim"] * rel["pagerank"]),
                "Description": desc
            })
            # 减少可用token数
            max_token -= num_tokens_from_string(str(relas[-1]))
            # 如果token数不足，删除最后一个关系并退出循环
            if max_token <= 0:
                relas = relas[:-1]
                break

        # 如果有实体，格式化实体信息
        if ents:
            ents = "\n---- Entities ----\n{}".format(pd.DataFrame(ents).to_csv())
        else:
            ents = ""
        # 如果有关系，格式化关系信息
        if relas:
            relas = "\n---- Relations ----\n{}".format(pd.DataFrame(relas).to_csv())
        else:
            relas = ""

        # 返回检索结果
        return {
                "chunk_id": get_uuid(),
                "content_ltks": "",
                "content_with_weight": ents + relas + self._community_retrival_([n for n, _ in ents_from_query], filters, kb_ids, idxnms,
                                                        comm_topn, max_token),
                "doc_id": "",
                "docnm_kwd": "Related content in Knowledge Graph",
                "kb_id": kb_ids,
                "important_kwd": [],
                "image_id": "",
                "similarity": 1.,
                "vector_similarity": 1.,
                "term_similarity": 0,
                "vector": [],
                "positions": [],
            }

    # 社区检索方法
    def _community_retrival_(self, entities, condition, kb_ids, idxnms, topn, max_token):
        ## 社区检索
        # 定义需要获取的字段
        fields = ["docnm_kwd", "content_with_weight"]
        # 创建排序表达式
        odr = OrderByExpr()
        # 按权重降序排序
        odr.desc("weight_flt")
        # 深拷贝过滤条件
        fltr = deepcopy(condition)
        # 设置知识图谱关键词为社区报告
        fltr["knowledge_graph_kwd"] = "community_report"
        # 设置实体关键词
        fltr["entities_kwd"] = entities
        # 执行搜索
        comm_res = self.dataStore.search(fields, [], fltr, [],
                                         OrderByExpr(), 0, topn, idxnms, kb_ids)
        # 获取字段
        comm_res_fields = self.dataStore.getFields(comm_res, fields)
        # 初始化文本列表
        txts = []
        # 遍历搜索结果
        for ii, (_, row) in enumerate(comm_res_fields.items()):
            # 解析内容
            obj = json.loads(row["content_with_weight"])
            # 格式化文本
            txts.append("# {}. {}\n## Content\n{}\n## Evidences\n{}\n".format(
                ii + 1, row["docnm_kwd"], obj["report"], obj["evidences"]))
            # 减少可用token数
            max_token -= num_tokens_from_string(str(txts[-1]))

        # 如果没有文本，返回空字符串
        if not txts:
            return ""
        # 返回格式化的社区报告
        return "\n---- Community Report ----\n" + "\n".join(txts)


# 主程序入口
if __name__ == "__main__":
    # 导入设置模块
    from api import settings
    # 导入参数解析模块
    import argparse
    # 导入LLM类型
    from api.db import LLMType
    # 导入知识库服务
    from api.db.services.knowledgebase_service import KnowledgebaseService
    # 导入LLM服务
    from api.db.services.llm_service import LLMBundle
    # 导入用户服务
    from api.db.services.user_service import TenantService
    # 导入搜索模块
    from rag.nlp import search

    # 初始化设置
    settings.init_settings()
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加租户ID参数
    parser.add_argument('-t', '--tenant_id', default=False, help="Tenant ID", action='store', required=True)
    # 添加知识库ID参数
    parser.add_argument('-d', '--kb_id', default=False, help="Knowledge base ID", action='store', required=True)
    # 添加问题参数
    parser.add_argument('-q', '--question', default=False, help="Question", action='store', required=True)
    # 解析参数
    args = parser.parse_args()

    # 获取知识库ID
    kb_id = args.kb_id
    # 获取租户信息
    _, tenant = TenantService.get_by_id(args.tenant_id)
    # 创建LLM包
    llm_bdl = LLMBundle(args.tenant_id, LLMType.CHAT, tenant.llm_id)
    # 获取知识库信息
    _, kb = KnowledgebaseService.get_by_id(kb_id)
    # 创建嵌入模型包
    embed_bdl = LLMBundle(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)

    # 创建知识图谱搜索对象
    kg = KGSearch(settings.docStoreConn)
    # 执行检索并打印结果
    print(kg.retrieval({"question": args.question, "kb_ids": [kb_id]},
                    search.index_name(kb.tenant_id), [kb_id], embed_bdl, llm_bdl))
