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
# 导入日志模块
import logging
# 导入函数式编程工具
from functools import partial
# 导入网络图处理库
import networkx as nx
# 导入异步处理库
import trio

# 导入设置模块
from api import settings
# 导入轻量级知识图谱提取器
from graphrag.light.graph_extractor import GraphExtractor as LightKGExt
# 导入通用知识图谱提取器
from graphrag.general.graph_extractor import GraphExtractor as GeneralKGExt
# 导入社区报告提取器
from graphrag.general.community_reports_extractor import CommunityReportsExtractor
# 导入实体解析模块
from graphrag.entity_resolution import EntityResolution
# 导入提取器基类
from graphrag.general.extractor import Extractor
# 导入图处理工具函数
from graphrag.utils import (
    graph_merge,
    set_entity,
    get_relation,
    set_relation,
    get_entity,
    get_graph,
    set_graph,
    chunk_id,
    update_nodes_pagerank_nhop_neighbour,
    does_graph_contains,
    get_graph_doc_ids,
)
# 导入NLP处理工具
from rag.nlp import rag_tokenizer, search
# 导入Redis连接
from rag.utils.redis_conn import REDIS_CONN


# 设置GraphRAG任务信息到Redis
def graphrag_task_set(tenant_id, kb_id, doc_id) -> bool:
    # 构建Redis键名
    key = f"graphrag:{tenant_id}:{kb_id}"
    # 设置键值对，过期时间为24小时
    ok = REDIS_CONN.set(key, doc_id, exp=3600 * 24)
    # 如果设置失败，抛出异常
    if not ok:
        raise Exception(f"Faild to set the {key} to {doc_id}")


# 从Redis获取GraphRAG任务信息
def graphrag_task_get(tenant_id, kb_id) -> str | None:
    # 构建Redis键名
    key = f"graphrag:{tenant_id}:{kb_id}"
    # 获取文档ID
    doc_id = REDIS_CONN.get(key)
    # 返回文档ID
    return doc_id


# 运行GraphRAG主函数
async def run_graphrag(
    row: dict,
    language,
    with_resolution: bool,
    with_community: bool,
    chat_model,
    embedding_model,
    callback,
):
    # 记录开始时间
    start = trio.current_time()
    # 从输入行中提取租户ID、知识库ID和文档ID
    tenant_id, kb_id, doc_id = row["tenant_id"], str(row["kb_id"]), row["doc_id"]
    # 初始化文本块列表
    chunks = []
    # 从检索器获取文档分块
    for d in settings.retrievaler.chunk_list(
        doc_id, tenant_id, [kb_id], fields=["content_with_weight", "doc_id"]
    ):
        # 添加文本内容到块列表
        chunks.append(d["content_with_weight"])

    # 更新图结构
    graph, doc_ids = await update_graph(
        # 根据配置选择图提取器类型
        LightKGExt
        if row["parser_config"]["graphrag"]["method"] != "general"
        else GeneralKGExt,
        tenant_id,
        kb_id,
        doc_id,
        chunks,
        language,
        row["parser_config"]["graphrag"]["entity_types"],
        chat_model,
        embedding_model,
        callback,
    )
    # 如果图为空，直接返回
    if not graph:
        return
    # 如果需要实体解析或社区检测，设置任务信息
    if with_resolution or with_community:
        graphrag_task_set(tenant_id, kb_id, doc_id)
    # 如果需要实体解析，执行解析
    if with_resolution:
        await resolve_entities(
            graph,
            doc_ids,
            tenant_id,
            kb_id,
            doc_id,
            chat_model,
            embedding_model,
            callback,
        )
    # 如果需要社区检测，执行检测
    if with_community:
        await extract_community(
            graph,
            doc_ids,
            tenant_id,
            kb_id,
            doc_id,
            chat_model,
            embedding_model,
            callback,
        )
    # 记录当前时间
    now = trio.current_time()
    # 回调通知任务完成
    callback(msg=f"GraphRAG for doc {doc_id} done in {now - start:.2f} seconds.")
    # 返回
    return


# 更新图结构函数
async def update_graph(
    extractor: Extractor,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    chunks: list[str],
    language,
    entity_types,
    llm_bdl,
    embed_bdl,
    callback,
):
    # 检查图是否已包含该文档
    contains = await does_graph_contains(tenant_id, kb_id, doc_id)
    # 如果已包含，取消操作
    if contains:
        callback(msg=f"Graph already contains {doc_id}, cancel myself")
        return None, None
    # 记录开始时间
    start = trio.current_time()
    # 创建提取器实例
    ext = extractor(
        llm_bdl,
        language=language,
        entity_types=entity_types,
        get_entity=partial(get_entity, tenant_id, kb_id),
        set_entity=partial(set_entity, tenant_id, kb_id, embed_bdl),
        get_relation=partial(get_relation, tenant_id, kb_id),
        set_relation=partial(set_relation, tenant_id, kb_id, embed_bdl),
    )
    # 提取实体和关系
    ents, rels = await ext(doc_id, chunks, callback)
    # 创建子图
    subgraph = nx.Graph()
    # 添加实体节点
    for en in ents:
        subgraph.add_node(en["entity_name"], entity_type=en["entity_type"])

    # 添加关系边
    for rel in rels:
        subgraph.add_edge(
            rel["src_id"],
            rel["tgt_id"],
            weight=rel["weight"],
            # description=rel["description"]
        )
    # 创建子图块
    chunk = {
        "content_with_weight": json.dumps(
            nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False, indent=2
        ),
        "knowledge_graph_kwd": "subgraph",
        "kb_id": kb_id,
        "source_id": [doc_id],
        "available_int": 0,
        "removed_kwd": "N",
    }
    # 生成块ID
    cid = chunk_id(chunk)
    # 异步插入子图到文档存储
    await trio.to_thread.run_sync(
        lambda: settings.docStoreConn.insert(
            [{"id": cid, **chunk}], search.index_name(tenant_id), kb_id
        )
    )
    # 记录当前时间
    now = trio.current_time()
    # 回调通知子图生成完成
    callback(msg=f"generated subgraph for doc {doc_id} in {now - start:.2f} seconds.")
    # 更新开始时间
    start = now

    # 循环合并图直到成功
    while True:
        # 初始化新图为子图
        new_graph = subgraph
        # 初始化当前文档ID集合
        now_docids = set([doc_id])
        # 获取旧图和旧文档ID
        old_graph, old_doc_ids = await get_graph(tenant_id, kb_id)
        # 如果旧图存在，合并图
        if old_graph is not None:
            logging.info("Merge with an exiting graph...................")
            new_graph = graph_merge(old_graph, subgraph)
        # 更新节点PageRank和邻居
        await update_nodes_pagerank_nhop_neighbour(tenant_id, kb_id, new_graph, 2)
        # 合并文档ID集合
        if old_doc_ids:
            for old_doc_id in old_doc_ids:
                now_docids.add(old_doc_id)
        # 获取最新的文档ID集合
        old_doc_ids2 = await get_graph_doc_ids(tenant_id, kb_id)
        # 计算文档ID差集
        delta_doc_ids = set(old_doc_ids2) - set(old_doc_ids)
        # 如果有差集，说明全局图已变化，需要重试
        if delta_doc_ids:
            callback(
                msg="The global graph has changed during merging, try again"
            )
            # 等待1秒后重试
            await trio.sleep(1)
            continue
        # 合并成功，跳出循环
        break
    # 设置新图和文档ID
    await set_graph(tenant_id, kb_id, new_graph, list(now_docids))
    # 记录当前时间
    now = trio.current_time()
    # 回调通知图合并完成
    callback(
        msg=f"merging subgraph for doc {doc_id} into the global graph done in {now - start:.2f} seconds."
    )
    # 返回新图和文档ID
    return new_graph, now_docids


# 解析实体函数
async def resolve_entities(
    graph,
    doc_ids,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    llm_bdl,
    embed_bdl,
    callback,
):
    # 获取当前工作的文档ID
    working_doc_id = graphrag_task_get(tenant_id, kb_id)
    # 如果当前文档不是工作文档，取消操作
    if doc_id != working_doc_id:
        callback(
            msg=f"Another graphrag task of doc_id {working_doc_id} is working on this kb, cancel myself"
        )
        return
    # 记录开始时间
    start = trio.current_time()
    # 创建实体解析器
    er = EntityResolution(
        llm_bdl,
        get_entity=partial(get_entity, tenant_id, kb_id),
        set_entity=partial(set_entity, tenant_id, kb_id, embed_bdl),
        get_relation=partial(get_relation, tenant_id, kb_id),
        set_relation=partial(set_relation, tenant_id, kb_id, embed_bdl),
    )
    # 执行实体解析
    reso = await er(graph, callback=callback)
    # 获取解析后的图
    graph = reso.graph
    # 回调通知移除的实体数量
    callback(msg=f"Graph resolution removed {len(reso.removed_entities)} nodes.")
    # 更新节点PageRank和邻居
    await update_nodes_pagerank_nhop_neighbour(tenant_id, kb_id, graph, 2)
    # 回调通知PageRank更新完成
    callback(msg="Graph resolution updated pagerank.")

    # 再次检查工作文档ID
    working_doc_id = graphrag_task_get(tenant_id, kb_id)
    # 如果当前文档不是工作文档，取消操作
    if doc_id != working_doc_id:
        callback(
            msg=f"Another graphrag task of doc_id {working_doc_id} is working on this kb, cancel myself"
        )
        return
    # 设置新图和文档ID
    await set_graph(tenant_id, kb_id, graph, doc_ids)

    # 删除被移除实体的关系（从实体出发）
    await trio.to_thread.run_sync(
        lambda: settings.docStoreConn.delete(
            {
                "knowledge_graph_kwd": "relation",
                "kb_id": kb_id,
                "from_entity_kwd": reso.removed_entities,
            },
            search.index_name(tenant_id),
            kb_id,
        )
    )
    # 删除被移除实体的关系（到实体）
    await trio.to_thread.run_sync(
        lambda: settings.docStoreConn.delete(
            {
                "knowledge_graph_kwd": "relation",
                "kb_id": kb_id,
                "to_entity_kwd": reso.removed_entities,
            },
            search.index_name(tenant_id),
            kb_id,
        )
    )
    # 删除被移除的实体
    await trio.to_thread.run_sync(
        lambda: settings.docStoreConn.delete(
            {
                "knowledge_graph_kwd": "entity",
                "kb_id": kb_id,
                "entity_kwd": reso.removed_entities,
            },
            search.index_name(tenant_id),
            kb_id,
        )
    )
    # 记录当前时间
    now = trio.current_time()
    # 回调通知实体解析完成
    callback(msg=f"Graph resolution done in {now - start:.2f}s.")


# 提取社区函数
async def extract_community(
    graph,
    doc_ids,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    llm_bdl,
    embed_bdl,
    callback,
):
    # 获取当前工作的文档ID
    working_doc_id = graphrag_task_get(tenant_id, kb_id)
    # 如果当前文档不是工作文档，取消操作
    if doc_id != working_doc_id:
        callback(
            msg=f"Another graphrag task of doc_id {working_doc_id} is working on this kb, cancel myself"
        )
        return
    # 记录开始时间
    start = trio.current_time()
    # 创建社区报告提取器
    ext = CommunityReportsExtractor(
        llm_bdl,
        get_entity=partial(get_entity, tenant_id, kb_id),
        set_entity=partial(set_entity, tenant_id, kb_id, embed_bdl),
        get_relation=partial(get_relation, tenant_id, kb_id),
        set_relation=partial(set_relation, tenant_id, kb_id, embed_bdl),
    )
    # 执行社区提取
    cr = await ext(graph, callback=callback)
    # 获取社区结构和报告
    community_structure = cr.structured_output
    community_reports = cr.output
    # 再次检查工作文档ID
    working_doc_id = graphrag_task_get(tenant_id, kb_id)
    # 如果当前文档不是工作文档，取消操作
    if doc_id != working_doc_id:
        callback(
            msg=f"Another graphrag task of doc_id {working_doc_id} is working on this kb, cancel myself"
        )
        return
    # 设置新图和文档ID
    await set_graph(tenant_id, kb_id, graph, doc_ids)

    # 记录当前时间
    now = trio.current_time()
    # 回调通知社区提取完成
    callback(
        msg=f"Graph extracted {len(cr.structured_output)} communities in {now - start:.2f}s."
    )
    # 更新开始时间
    start = now
    # 删除旧的社区报告
    await trio.to_thread.run_sync(
        lambda: settings.docStoreConn.delete(
            {"knowledge_graph_kwd": "community_report", "kb_id": kb_id},
            search.index_name(tenant_id),
            kb_id,
        )
    )
    # 遍历社区结构和报告
    for stru, rep in zip(community_structure, community_reports):
        # 创建报告对象
        obj = {
            "report": rep,
            "evidences": "\n".join([f["explanation"] for f in stru["findings"]]),
        }
        # 创建社区块
        chunk = {
            "docnm_kwd": stru["title"],
            "title_tks": rag_tokenizer.tokenize(stru["title"]),
            "content_with_weight": json.dumps(obj, ensure_ascii=False),
            "content_ltks": rag_tokenizer.tokenize(
                obj["report"] + " " + obj["evidences"]
            ),
            "knowledge_graph_kwd": "community_report",
            "weight_flt": stru["weight"],
            "entities_kwd": stru["entities"],
            "important_kwd": stru["entities"],
            "kb_id": kb_id,
            "source_id": doc_ids,
            "available_int": 0,
        }
        # 生成细粒度标记
        chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(
            chunk["content_ltks"]
        )
        # try:
        #    ebd, _ = embed_bdl.encode([", ".join(community["entities"])])
        #    chunk["q_%d_vec" % len(ebd[0])] = ebd[0]
        # except Exception as e:
        #    logging.exception(f"Fail to embed entity relation: {e}")
        # 异步插入社区报告到文档存储
        await trio.to_thread.run_sync(
            lambda: settings.docStoreConn.insert(
                [{"id": chunk_id(chunk), **chunk}], search.index_name(tenant_id)
            )
        )

    # 记录当前时间
    now = trio.current_time()
    # 回调通知社区索引完成
    callback(
        msg=f"Graph indexed {len(cr.structured_output)} communities in {now - start:.2f}s."
    )
    # 返回社区结构和报告
    return community_structure, community_reports
