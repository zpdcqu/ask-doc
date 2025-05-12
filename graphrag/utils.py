# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
 - [LightRag](https://github.com/HKUDS/LightRAG)
"""

import html  # 导入HTML处理模块
import json  # 导入JSON处理模块
import logging  # 导入日志模块
import re  # 导入正则表达式模块
import time  # 导入时间模块
from collections import defaultdict  # 导入默认字典
from copy import deepcopy  # 导入深拷贝函数
from hashlib import md5  # 导入MD5哈希函数
from typing import Any, Callable  # 导入类型提示
import os  # 导入操作系统模块
import trio  # 导入异步并发库

import networkx as nx  # 导入网络图处理库
import numpy as np  # 导入数值计算库
import xxhash  # 导入高性能哈希库
from networkx.readwrite import json_graph  # 导入图形JSON转换

from api import settings  # 导入API设置
from rag.nlp import search, rag_tokenizer  # 导入搜索和分词器
from rag.utils.doc_store_conn import OrderByExpr  # 导入排序表达式
from rag.utils.redis_conn import REDIS_CONN  # 导入Redis连接

ErrorHandlerFn = Callable[[BaseException | None, str | None, dict | None], None]  # 定义错误处理函数类型

chat_limiter = trio.CapacityLimiter(int(os.environ.get('MAX_CONCURRENT_CHATS', 10)))  # 创建聊天并发限制器

def perform_variable_replacements(
    input: str, history: list[dict] | None = None, variables: dict | None = None
) -> str:
    """Perform variable replacements on the input string and in a chat log."""
    # 在输入字符串和聊天记录中执行变量替换
    if history is None:
        history = []  # 如果历史为空，初始化为空列表
    if variables is None:
        variables = {}  # 如果变量为空，初始化为空字典
    result = input  # 初始化结果为输入

    def replace_all(input: str) -> str:
        result = input  # 初始化结果为输入
        for k, v in variables.items():
            result = result.replace(f"{{{k}}}", v)  # 替换所有变量占位符
        return result

    result = replace_all(result)  # 对结果执行替换
    for i, entry in enumerate(history):
        if entry.get("role") == "system":
            entry["content"] = replace_all(entry.get("content") or "")  # 对系统消息内容执行替换

    return result  # 返回替换后的结果


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # 清理输入字符串，移除HTML转义、控制字符和其他不需要的字符
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input  # 如果输入不是字符串，直接返回

    result = html.unescape(input.strip())  # 解除HTML转义并去除首尾空白
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\"\x00-\x1f\x7f-\x9f]", "", result)  # 移除控制字符和引号


def dict_has_keys_with_types(
    data: dict, expected_fields: list[tuple[str, type]]
) -> bool:
    """Return True if the given dictionary has the given keys with the given types."""
    # 检查字典是否包含指定键且值类型符合预期
    for field, field_type in expected_fields:
        if field not in data:
            return False  # 如果字段不存在，返回False

        value = data[field]
        if not isinstance(value, field_type):
            return False  # 如果字段类型不匹配，返回False
    return True  # 所有检查通过，返回True


def get_llm_cache(llmnm, txt, history, genconf):
    # 获取LLM缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(llmnm).encode("utf-8"))  # 添加模型名称到哈希
    hasher.update(str(txt).encode("utf-8"))  # 添加文本到哈希
    hasher.update(str(history).encode("utf-8"))  # 添加历史记录到哈希
    hasher.update(str(genconf).encode("utf-8"))  # 添加生成配置到哈希

    k = hasher.hexdigest()  # 获取哈希值
    bin = REDIS_CONN.get(k)  # 从Redis获取缓存
    if not bin:
        return  # 如果缓存不存在，返回None
    return bin  # 返回缓存内容


def set_llm_cache(llmnm, txt, v, history, genconf):
    # 设置LLM缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(llmnm).encode("utf-8"))  # 添加模型名称到哈希
    hasher.update(str(txt).encode("utf-8"))  # 添加文本到哈希
    hasher.update(str(history).encode("utf-8"))  # 添加历史记录到哈希
    hasher.update(str(genconf).encode("utf-8"))  # 添加生成配置到哈希

    k = hasher.hexdigest()  # 获取哈希值
    REDIS_CONN.set(k, v.encode("utf-8"), 24*3600)  # 设置缓存，有效期24小时


def get_embed_cache(llmnm, txt):
    # 获取嵌入缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(llmnm).encode("utf-8"))  # 添加模型名称到哈希
    hasher.update(str(txt).encode("utf-8"))  # 添加文本到哈希

    k = hasher.hexdigest()  # 获取哈希值
    bin = REDIS_CONN.get(k)  # 从Redis获取缓存
    if not bin:
        return  # 如果缓存不存在，返回None
    return np.array(json.loads(bin))  # 将JSON转换为NumPy数组并返回


def set_embed_cache(llmnm, txt, arr):
    # 设置嵌入缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(llmnm).encode("utf-8"))  # 添加模型名称到哈希
    hasher.update(str(txt).encode("utf-8"))  # 添加文本到哈希

    k = hasher.hexdigest()  # 获取哈希值
    arr = json.dumps(arr.tolist() if isinstance(arr, np.ndarray) else arr)  # 将数组转换为JSON
    REDIS_CONN.set(k, arr.encode("utf-8"), 24*3600)  # 设置缓存，有效期24小时


def get_tags_from_cache(kb_ids):
    # 从缓存获取标签
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(kb_ids).encode("utf-8"))  # 添加知识库ID到哈希

    k = hasher.hexdigest()  # 获取哈希值
    bin = REDIS_CONN.get(k)  # 从Redis获取缓存
    if not bin:
        return  # 如果缓存不存在，返回None
    return bin  # 返回缓存内容


def set_tags_to_cache(kb_ids, tags):
    # 设置标签到缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(kb_ids).encode("utf-8"))  # 添加知识库ID到哈希

    k = hasher.hexdigest()  # 获取哈希值
    REDIS_CONN.set(k, json.dumps(tags).encode("utf-8"), 600)  # 设置缓存，有效期10分钟


def graph_merge(g1, g2):
    # 合并两个图
    g = g2.copy()  # 复制第二个图
    for n, attr in g1.nodes(data=True):
        if n not in g2.nodes():
            g.add_node(n, **attr)  # 添加不存在的节点
            continue

    for source, target, attr in g1.edges(data=True):
        if g.has_edge(source, target):
            g[source][target].update({"weight": attr.get("weight", 0)+1})  # 更新边权重
            continue
        g.add_edge(source, target)  # 添加新边

    for node_degree in g.degree:
        g.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])  # 更新节点排名
    return g  # 返回合并后的图


def compute_args_hash(*args):
    # 计算参数哈希值
    return md5(str(args).encode()).hexdigest()  # 使用MD5计算哈希


def handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # 处理单个实体提取
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None  # 如果记录不符合要求，返回None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())  # 清理并转换实体名称为大写
    if not entity_name.strip():
        return None  # 如果实体名为空，返回None
    entity_type = clean_str(record_attributes[2].upper())  # 清理并转换实体类型为大写
    entity_description = clean_str(record_attributes[3])  # 清理实体描述
    entity_source_id = chunk_key  # 设置实体来源ID
    return dict(
        entity_name=entity_name.upper(),  # 实体名称（大写）
        entity_type=entity_type.upper(),  # 实体类型（大写）
        description=entity_description,  # 实体描述
        source_id=entity_source_id,  # 来源ID
    )


def handle_single_relationship_extraction(record_attributes: list[str], chunk_key: str):
    # 处理单个关系提取
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None  # 如果记录不符合要求，返回None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())  # 清理并转换源实体为大写
    target = clean_str(record_attributes[2].upper())  # 清理并转换目标实体为大写
    edge_description = clean_str(record_attributes[3])  # 清理边描述

    edge_keywords = clean_str(record_attributes[4])  # 清理边关键词
    edge_source_id = chunk_key  # 设置边来源ID
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )  # 设置边权重
    pair = sorted([source.upper(), target.upper()])  # 对源和目标实体排序
    return dict(
        src_id=pair[0],  # 源ID
        tgt_id=pair[1],  # 目标ID
        weight=weight,  # 权重
        description=edge_description,  # 描述
        keywords=edge_keywords,  # 关键词
        source_id=edge_source_id,  # 来源ID
        metadata={"created_at": time.time()},  # 元数据（创建时间）
    )


def pack_user_ass_to_openai_messages(*args: str):
    # 将用户和助手消息打包为OpenAI消息格式
    roles = ["user", "assistant"]  # 角色列表
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]  # 交替分配角色


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    # 通过多个标记分割字符串
    if not markers:
        return [content]  # 如果没有标记，返回原字符串
    results = re.split("|".join(re.escape(marker) for marker in markers), content)  # 使用正则表达式分割
    return [r.strip() for r in results if r.strip()]  # 返回非空结果


def is_float_regex(value):
    # 检查字符串是否为浮点数
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))  # 使用正则表达式匹配浮点数格式


def chunk_id(chunk):
    # 生成块ID
    return xxhash.xxh64((chunk["content_with_weight"] + chunk["kb_id"]).encode("utf-8")).hexdigest()  # 使用内容和知识库ID生成哈希

def get_entity_cache(tenant_id, kb_id, ent_name) -> str | list[str]:
    # 获取实体缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(tenant_id).encode("utf-8"))  # 添加租户ID到哈希
    hasher.update(str(kb_id).encode("utf-8"))  # 添加知识库ID到哈希
    hasher.update(str(ent_name).encode("utf-8"))  # 添加实体名称到哈希

    k = hasher.hexdigest()  # 获取哈希值
    bin = REDIS_CONN.get(k)  # 从Redis获取缓存
    if not bin:
        return  # 如果缓存不存在，返回None
    return json.loads(bin)  # 返回解析后的JSON


def set_entity_cache(tenant_id, kb_id, ent_name, content_with_weight):
    # 设置实体缓存
    hasher = xxhash.xxh64()  # 创建哈希对象
    hasher.update(str(tenant_id).encode("utf-8"))  # 添加租户ID到哈希
    hasher.update(str(kb_id).encode("utf-8"))  # 添加知识库ID到哈希
    hasher.update(str(ent_name).encode("utf-8"))  # 添加实体名称到哈希

    k = hasher.hexdigest()  # 获取哈希值
    REDIS_CONN.set(k, content_with_weight.encode("utf-8"), 3600)  # 设置缓存，有效期1小时


def get_entity(tenant_id, kb_id, ent_name):
    # 获取实体
    cache = get_entity_cache(tenant_id, kb_id, ent_name)  # 尝试从缓存获取
    if cache:
        return cache  # 如果缓存存在，返回缓存
    conds = {
        "fields": ["content_with_weight"],
        "entity_kwd": ent_name,
        "size": 10000,
        "knowledge_graph_kwd": ["entity"]
    }  # 设置搜索条件
    res = []
    es_res = settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id])  # 执行搜索
    for id in es_res.ids:
        try:
            if isinstance(ent_name, str):
                set_entity_cache(tenant_id, kb_id, ent_name, es_res.field[id]["content_with_weight"])  # 设置缓存
                return json.loads(es_res.field[id]["content_with_weight"])  # 返回解析后的JSON
            res.append(json.loads(es_res.field[id]["content_with_weight"]))  # 添加到结果列表
        except Exception:
            continue  # 忽略异常

    return res  # 返回结果列表


def set_entity(tenant_id, kb_id, embd_mdl, ent_name, meta):
    # 设置实体
    chunk = {
        "important_kwd": [ent_name],  # 重要关键词
        "title_tks": rag_tokenizer.tokenize(ent_name),  # 标题分词
        "entity_kwd": ent_name,  # 实体关键词
        "knowledge_graph_kwd": "entity",  # 知识图谱关键词
        "entity_type_kwd": meta["entity_type"],  # 实体类型关键词
        "content_with_weight": json.dumps(meta, ensure_ascii=False),  # 带权重的内容
        "content_ltks": rag_tokenizer.tokenize(meta["description"]),  # 内容分词
        "source_id": list(set(meta["source_id"])),  # 来源ID
        "kb_id": kb_id,  # 知识库ID
        "available_int": 0  # 可用整数
    }
    chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])  # 细粒度分词
    set_entity_cache(tenant_id, kb_id, ent_name, chunk["content_with_weight"])  # 设置实体缓存
    res = settings.retrievaler.search({"entity_kwd": ent_name, "size": 1, "fields": []},
                                      search.index_name(tenant_id), [kb_id])  # 搜索实体
    if res.ids:
        settings.docStoreConn.update({"entity_kwd": ent_name}, chunk, search.index_name(tenant_id), kb_id)  # 更新实体
    else:
        ebd = get_embed_cache(embd_mdl.llm_name, ent_name)  # 获取嵌入缓存
        if ebd is None:
            try:
                ebd, _ = embd_mdl.encode([ent_name])  # 编码实体名称
                ebd = ebd[0]  # 获取第一个嵌入
                set_embed_cache(embd_mdl.llm_name, ent_name, ebd)  # 设置嵌入缓存
            except Exception as e:
                logging.exception(f"Fail to embed entity: {e}")  # 记录异常
        if ebd is not None:
            chunk["q_%d_vec" % len(ebd)] = ebd  # 设置向量
        settings.docStoreConn.insert([{"id": chunk_id(chunk), **chunk}], search.index_name(tenant_id), kb_id)  # 插入实体


def get_relation(tenant_id, kb_id, from_ent_name, to_ent_name, size=1):
    # 获取关系
    ents = from_ent_name
    if isinstance(ents, str):
        ents = [from_ent_name]  # 如果是字符串，转换为列表
    if isinstance(to_ent_name, str):
        to_ent_name = [to_ent_name]  # 如果是字符串，转换为列表
    ents.extend(to_ent_name)  # 合并实体列表
    ents = list(set(ents))  # 去重
    conds = {
        "fields": ["content_with_weight"],
        "size": size,
        "from_entity_kwd": ents,
        "to_entity_kwd": ents,
        "knowledge_graph_kwd": ["relation"]
    }  # 设置搜索条件
    res = []
    es_res = settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id] if isinstance(kb_id, str) else kb_id)  # 执行搜索
    for id in es_res.ids:
        try:
            if size == 1:
                return json.loads(es_res.field[id]["content_with_weight"])  # 返回解析后的JSON
            res.append(json.loads(es_res.field[id]["content_with_weight"]))  # 添加到结果列表
        except Exception:
            continue  # 忽略异常
    return res  # 返回结果列表


def set_relation(tenant_id, kb_id, embd_mdl, from_ent_name, to_ent_name, meta):
    # 设置关系
    chunk = {
        "from_entity_kwd": from_ent_name,  # 源实体关键词
        "to_entity_kwd": to_ent_name,  # 目标实体关键词
        "knowledge_graph_kwd": "relation",  # 知识图谱关键词
        "content_with_weight": json.dumps(meta, ensure_ascii=False),  # 带权重的内容
        "content_ltks": rag_tokenizer.tokenize(meta["description"]),  # 内容分词
        "important_kwd": meta["keywords"],  # 重要关键词
        "source_id": list(set(meta["source_id"])),  # 来源ID
        "weight_int": int(meta["weight"]),  # 权重整数
        "kb_id": kb_id,  # 知识库ID
        "available_int": 0  # 可用整数
    }
    chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])  # 细粒度分词
    res = settings.retrievaler.search({"from_entity_kwd": to_ent_name, "to_entity_kwd": to_ent_name, "size": 1, "fields": []},
                                      search.index_name(tenant_id), [kb_id])  # 搜索关系

    if res.ids:
        settings.docStoreConn.update({"from_entity_kwd": from_ent_name, "to_entity_kwd": to_ent_name},
                                 chunk,
                                 search.index_name(tenant_id), kb_id)  # 更新关系
    else:
        txt = f"{from_ent_name}->{to_ent_name}"  # 创建文本表示
        ebd = get_embed_cache(embd_mdl.llm_name, txt)  # 获取嵌入缓存
        if ebd is None:
            try:
                ebd, _ = embd_mdl.encode([txt+f": {meta['description']}"])  # 编码关系文本
                ebd = ebd[0]  # 获取第一个嵌入
                set_embed_cache(embd_mdl.llm_name, txt, ebd)  # 设置嵌入缓存
            except Exception as e:
                logging.exception(f"Fail to embed entity relation: {e}")  # 记录异常
        if ebd is not None:
            chunk["q_%d_vec" % len(ebd)] = ebd  # 设置向量
        settings.docStoreConn.insert([{"id": chunk_id(chunk), **chunk}], search.index_name(tenant_id), kb_id)  # 插入关系

async def does_graph_contains(tenant_id, kb_id, doc_id):
    # 检查图是否包含文档ID
    # Get doc_ids of graph
    fields = ["source_id"]
    condition = {
        "knowledge_graph_kwd": ["graph"],
        "removed_kwd": "N",
    }
    res = await trio.to_thread.run_sync(lambda: settings.docStoreConn.search(fields, [], condition, [], OrderByExpr(), 0, 1, search.index_name(tenant_id), [kb_id]))  # 异步搜索
    fields2 = settings.docStoreConn.getFields(res, fields)  # 获取字段
    graph_doc_ids = set()
    for chunk_id in fields2.keys():
        graph_doc_ids = set(fields2[chunk_id]["source_id"])  # 获取图文档ID
    return doc_id in graph_doc_ids  # 检查文档ID是否在图中

async def get_graph_doc_ids(tenant_id, kb_id) -> list[str]:
    # 获取图文档ID
    conds = {
        "fields": ["source_id"],
        "removed_kwd": "N",
        "size": 1,
        "knowledge_graph_kwd": ["graph"]
    }  # 设置搜索条件
    res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id]))  # 异步搜索
    doc_ids = []
    if res.total == 0:
        return doc_ids  # 如果没有结果，返回空列表
    for id in res.ids:
        doc_ids = res.field[id]["source_id"]  # 获取文档ID
    return doc_ids  # 返回文档ID列表


async def get_graph(tenant_id, kb_id):
    # 获取图
    conds = {
        "fields": ["content_with_weight", "source_id"],
        "removed_kwd": "N",
        "size": 1,
        "knowledge_graph_kwd": ["graph"]
    }  # 设置搜索条件
    res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search(conds, search.index_name(tenant_id), [kb_id]))  # 异步搜索
    if res.total == 0:
        return None, []  # 如果没有结果，返回None和空列表
    for id in res.ids:
        try:
            return json_graph.node_link_graph(json.loads(res.field[id]["content_with_weight"]), edges="edges"), \
                   res.field[id]["source_id"]  # 返回图和文档ID
        except Exception:
            continue  # 忽略异常
    result = await rebuild_graph(tenant_id, kb_id)  # 重建图
    return result  # 返回重建结果


async def set_graph(tenant_id, kb_id, graph, docids):
    # 设置图
    chunk = {
        "content_with_weight": json.dumps(nx.node_link_data(graph, edges="edges"), ensure_ascii=False,
                                          indent=2),  # 带权重的内容
        "knowledge_graph_kwd": "graph",  # 知识图谱关键词
        "kb_id": kb_id,  # 知识库ID
        "source_id": list(docids),  # 来源ID
        "available_int": 0,  # 可用整数
        "removed_kwd": "N"  # 移除关键词
    }     
    res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search({"knowledge_graph_kwd": "graph", "size": 1, "fields": []}, search.index_name(tenant_id), [kb_id]))  # 异步搜索
    if res.ids:
        await trio.to_thread.run_sync(lambda: settings.docStoreConn.update({"knowledge_graph_kwd": "graph"}, chunk,
                                     search.index_name(tenant_id), kb_id))  # 更新图
    else:
        await trio.to_thread.run_sync(lambda: settings.docStoreConn.insert([{"id": chunk_id(chunk), **chunk}], search.index_name(tenant_id), kb_id))  # 插入图


def is_continuous_subsequence(subseq, seq):
    # 检查是否为连续子序列
    def find_all_indexes(tup, value):
        indexes = []
        start = 0
        while True:
            try:
                index = tup.index(value, start)  # 查找值的索引
                indexes.append(index)  # 添加到索引列表
                start = index + 1  # 更新起始位置
            except ValueError:
                break  # 找不到更多值，退出循环
        return indexes  # 返回索引列表

    index_list = find_all_indexes(seq,subseq[0])  # 查找第一个元素的所有索引
    for idx in index_list:
        if idx!=len(seq)-1:
            if seq[idx+1]==subseq[-1]:
                return True
    return False


def merge_tuples(list1, list2):
    result = []
    for tup in list1:
        last_element = tup[-1]
        if last_element in tup[:-1]:
            result.append(tup)
        else:
            matching_tuples = [t for t in list2 if t[0] == last_element]
            already_match_flag = 0
            for match in matching_tuples:
                matchh = (match[1], match[0])
                if is_continuous_subsequence(match, tup) or is_continuous_subsequence(matchh, tup):
                    continue
                already_match_flag = 1
                merged_tuple = tup + match[1:]
                result.append(merged_tuple)
            if not already_match_flag:
                result.append(tup)
    return result


async def update_nodes_pagerank_nhop_neighbour(tenant_id, kb_id, graph, n_hop):
    def n_neighbor(id):
        nonlocal graph, n_hop
        count = 0
        source_edge = list(graph.edges(id))
        if not source_edge:
            return []
        count = count + 1
        while count < n_hop:
            count = count + 1
            sc_edge = deepcopy(source_edge)
            source_edge = []
            for pair in sc_edge:
                append_edge = list(graph.edges(pair[-1]))
                for tuples in merge_tuples([pair], append_edge):
                    source_edge.append(tuples)
        nbrs = []
        for path in source_edge:
            n = {"path": path, "weights": []}
            wts = nx.get_edge_attributes(graph, 'weight')
            for i in range(len(path)-1):
                f, t = path[i], path[i+1]
                n["weights"].append(wts.get((f, t), 0))
            nbrs.append(n)
        return nbrs

    pr = nx.pagerank(graph)
    try:
        async with trio.open_nursery() as nursery:
            for n, p in pr.items():
                graph.nodes[n]["pagerank"] = p
                nursery.start_soon(lambda: trio.to_thread.run_sync(lambda: settings.docStoreConn.update({"entity_kwd": n, "kb_id": kb_id},
                                                {"rank_flt": p,
                                                "n_hop_with_weight": json.dumps((n), ensure_ascii=False)},
                                                search.index_name(tenant_id), kb_id)))
    except Exception as e:
        logging.exception(e)

    ty2ents = defaultdict(list)
    for p, r in sorted(pr.items(), key=lambda x: x[1], reverse=True):
        ty = graph.nodes[p].get("entity_type")
        if not ty or len(ty2ents[ty]) > 12:
            continue
        ty2ents[ty].append(p)

    chunk = {
        "content_with_weight": json.dumps(ty2ents, ensure_ascii=False),
        "kb_id": kb_id,
        "knowledge_graph_kwd": "ty2ents",
        "available_int": 0
    }
    res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search({"knowledge_graph_kwd": "ty2ents", "size": 1, "fields": []},
                                      search.index_name(tenant_id), [kb_id]))
    if res.ids:
        await trio.to_thread.run_sync(lambda: settings.docStoreConn.update({"knowledge_graph_kwd": "ty2ents"},
                                     chunk,
                                     search.index_name(tenant_id), kb_id))
    else:
        await trio.to_thread.run_sync(lambda: settings.docStoreConn.insert([{"id": chunk_id(chunk), **chunk}], search.index_name(tenant_id), kb_id))


async def get_entity_type2sampels(idxnms, kb_ids: list):
    es_res = await trio.to_thread.run_sync(lambda: settings.retrievaler.search({"knowledge_graph_kwd": "ty2ents", "kb_id": kb_ids,
                                       "size": 10000,
                                       "fields": ["content_with_weight"]},
                                      idxnms, kb_ids))

    res = defaultdict(list)
    for id in es_res.ids:
        smp = es_res.field[id].get("content_with_weight")
        if not smp:
            continue
        try:
            smp = json.loads(smp)
        except Exception as e:
            logging.exception(e)

        for ty, ents in smp.items():
            res[ty].extend(ents)
    return res


def flat_uniq_list(arr, key):
    res = []
    for a in arr:
        a = a[key]
        if isinstance(a, list):
            res.extend(a)
        else:
            res.append(a)
    return list(set(res))


async def rebuild_graph(tenant_id, kb_id):
    graph = nx.Graph()
    src_ids = []
    flds = ["entity_kwd", "entity_type_kwd", "from_entity_kwd", "to_entity_kwd", "weight_int", "knowledge_graph_kwd", "source_id"]
    bs = 256
    for i in range(0, 39*bs, bs):
        es_res = await trio.to_thread.run_sync(lambda: settings.docStoreConn.search(flds, [],
                                 {"kb_id": kb_id, "knowledge_graph_kwd": ["entity", "relation"]},
                                 [],
                                 OrderByExpr(),
                                 i, bs, search.index_name(tenant_id), [kb_id]
                                 ))
        tot = settings.docStoreConn.getTotal(es_res)
        if tot == 0:
            return None, None

        es_res = settings.docStoreConn.getFields(es_res, flds)
        for id, d in es_res.items():
            src_ids.extend(d.get("source_id", []))
            if d["knowledge_graph_kwd"] == "entity":
                graph.add_node(d["entity_kwd"], entity_type=d["entity_type_kwd"])
            elif "from_entity_kwd" in d and "to_entity_kwd" in d:
                graph.add_edge(
                    d["from_entity_kwd"],
                    d["to_entity_kwd"],
                    weight=int(d["weight_int"])
                )

        if len(es_res.keys()) < 128:
            return graph, list(set(src_ids))

    return graph, list(set(src_ids))
