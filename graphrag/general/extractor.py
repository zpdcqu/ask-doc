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
# 导入日志模块
import logging
# 导入正则表达式模块
import re
# 导入集合数据类型
from collections import defaultdict, Counter
# 导入深拷贝函数
from copy import deepcopy
# 导入类型注解
from typing import Callable
# 导入异步处理库
import trio

# 导入描述摘要提示模板
from graphrag.general.graph_prompt import SUMMARIZE_DESCRIPTIONS_PROMPT
# 导入工具函数
from graphrag.utils import get_llm_cache, set_llm_cache, handle_single_entity_extraction, \
    handle_single_relationship_extraction, split_string_by_multi_markers, flat_uniq_list, chat_limiter
# 导入聊天模型基类
from rag.llm.chat_model import Base as CompletionLLM
# 导入消息处理函数
from rag.prompts import message_fit_in
# 导入文本截断函数
from rag.utils import truncate

# 定义图字段分隔符
GRAPH_FIELD_SEP = "<SEP>"
# 定义默认实体类型列表
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
# 定义实体提取最大尝试次数
ENTITY_EXTRACTION_MAX_GLEANINGS = 2


# 定义提取器基类
class Extractor:
    # 声明LLM调用器属性
    _llm: CompletionLLM

    # 初始化方法
    def __init__(
        self,
        llm_invoker: CompletionLLM,
        language: str | None = "English",
        entity_types: list[str] | None = None,
        get_entity: Callable | None = None,
        set_entity: Callable | None = None,
        get_relation: Callable | None = None,
        set_relation: Callable | None = None,
    ):
        # 设置LLM调用器
        self._llm = llm_invoker
        # 设置语言
        self._language = language
        # 设置实体类型，如果未提供则使用默认值
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES
        # 设置获取实体的函数
        self._get_entity_ = get_entity
        # 设置设置实体的函数
        self._set_entity_ = set_entity
        # 设置获取关系的函数
        self._get_relation_ = get_relation
        # 设置设置关系的函数
        self._set_relation_ = set_relation

    # 聊天方法，用于调用LLM
    def _chat(self, system, history, gen_conf):
        # 深拷贝历史记录
        hist = deepcopy(history)
        # 深拷贝生成配置
        conf = deepcopy(gen_conf)
        # 尝试从缓存获取响应
        response = get_llm_cache(self._llm.llm_name, system, hist, conf)
        # 如果缓存中有响应，直接返回
        if response:
            return response
        # 确保系统消息适合LLM的最大长度
        _, system_msg = message_fit_in([{"role": "system", "content": system}], int(self._llm.max_length * 0.97))
        # 调用LLM获取响应
        response = self._llm.chat(system_msg[0]["content"], hist, conf)
        # 移除思考过程
        response = re.sub(r"<think>.*</think>", "", response, flags=re.DOTALL)
        # 如果响应中包含错误标记，抛出异常
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        # 将响应缓存
        set_llm_cache(self._llm.llm_name, system, response, history, gen_conf)
        # 返回响应
        return response

    # 处理实体和关系的方法
    def _entities_and_relations(self, chunk_key: str, records: list, tuple_delimiter: str):
        # 初始化可能的节点字典
        maybe_nodes = defaultdict(list)
        # 初始化可能的边字典
        maybe_edges = defaultdict(list)
        # 将实体类型转换为小写
        ent_types = [t.lower() for t in self._entity_types]
        # 遍历记录
        for record in records:
            # 按元组分隔符分割记录属性
            record_attributes = split_string_by_multi_markers(
                record, [tuple_delimiter]
            )

            # 尝试提取实体
            if_entities = handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            # 如果成功提取实体且实体类型在预定义类型中，添加到节点字典
            if if_entities is not None and if_entities.get("entity_type", "unknown").lower() in ent_types:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # 尝试提取关系
            if_relation = handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            # 如果成功提取关系，添加到边字典
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        # 返回节点和边字典
        return dict(maybe_nodes), dict(maybe_edges)

    # 主调用方法
    async def __call__(
        self, doc_id: str, chunks: list[str],
            callback: Callable | None = None
    ):
        # 设置回调函数
        self.callback = callback
        # 记录开始时间
        start_ts = trio.current_time()
        # 初始化结果列表
        out_results = []
        # 创建异步任务组
        async with trio.open_nursery() as nursery:
            # 遍历文本块
            for i, ck in enumerate(chunks):
                # 截断文本块以适应LLM最大长度
                ck = truncate(ck, int(self._llm.max_length*0.8))
                # 启动异步任务处理单个内容块
                nursery.start_soon(lambda: self._process_single_content((doc_id, ck), i, len(chunks), out_results))

        # 初始化可能的节点字典
        maybe_nodes = defaultdict(list)
        # 初始化可能的边字典
        maybe_edges = defaultdict(list)
        # 初始化令牌计数
        sum_token_count = 0
        # 处理结果
        for m_nodes, m_edges, token_count in out_results:
            # 合并节点
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            # 合并边，确保边的顺序一致
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
            # 累加令牌计数
            sum_token_count += token_count
        # 记录当前时间
        now = trio.current_time()
        # 如果有回调函数，调用它报告进度
        if callback:
            callback(msg = f"Entities and relationships extraction done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {sum_token_count} tokens, {now-start_ts:.2f}s.")
        # 更新开始时间
        start_ts = now
        # 记录实体合并开始
        logging.info("Entities merging...")
        # 初始化所有实体数据列表
        all_entities_data = []
        # 创建异步任务组
        async with trio.open_nursery() as nursery:
            # 遍历节点
            for en_nm, ents in maybe_nodes.items():
                # 启动异步任务合并节点
                nursery.start_soon(lambda: self._merge_nodes(en_nm, ents, all_entities_data))
        # 记录当前时间
        now = trio.current_time()
        # 如果有回调函数，调用它报告进度
        if callback:
            callback(msg = f"Entities merging done, {now-start_ts:.2f}s.")

        # 更新开始时间
        start_ts = now
        # 记录关系合并开始
        logging.info("Relationships merging...")
        # 初始化所有关系数据列表
        all_relationships_data = []
        # 创建异步任务组
        async with trio.open_nursery() as nursery:
            # 遍历边
            for (src, tgt), rels in maybe_edges.items():
                # 启动异步任务合并边
                nursery.start_soon(lambda: self._merge_edges(src, tgt, rels, all_relationships_data))
        # 记录当前时间
        now = trio.current_time()
        # 如果有回调函数，调用它报告进度
        if callback:
            callback(msg = f"Relationships merging done, {now-start_ts:.2f}s.")

        # 检查是否提取到实体和关系
        if not len(all_entities_data) and not len(all_relationships_data):
            logging.warning(
                "Didn't extract any entities and relationships, maybe your LLM is not working"
            )

        # 检查是否提取到实体
        if not len(all_entities_data):
            logging.warning("Didn't extract any entities")
        # 检查是否提取到关系
        if not len(all_relationships_data):
            logging.warning("Didn't extract any relationships")

        # 返回所有实体和关系数据
        return all_entities_data, all_relationships_data

    # 合并节点的异步方法
    async def _merge_nodes(self, entity_name: str, entities: list[dict], all_relationships_data):
        # 如果实体列表为空，直接返回
        if not entities:
            return
        # 初始化已有实体类型列表
        already_entity_types = []
        # 初始化已有源ID列表
        already_source_ids = []
        # 初始化已有描述列表
        already_description = []

        # 尝试获取已有节点
        already_node = self._get_entity_(entity_name)
        # 如果已有节点存在
        if already_node:
            # 添加已有实体类型
            already_entity_types.append(already_node["entity_type"])
            # 添加已有源ID
            already_source_ids.extend(already_node["source_id"])
            # 添加已有描述
            already_description.append(already_node["description"])

        # 确定实体类型，选择出现频率最高的类型
        entity_type = sorted(
            Counter(
                [dp["entity_type"] for dp in entities] + already_entity_types
            ).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        # 合并描述，使用分隔符连接并去重
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in entities] + already_description))
        )
        # 获取唯一的源ID列表
        already_source_ids = flat_uniq_list(entities, "source_id")
        # 处理实体描述摘要
        description = await self._handle_entity_relation_summary(entity_name, description)
        # 构建节点数据
        node_data = dict(
            entity_type=entity_type,
            description=description,
            source_id=already_source_ids,
        )
        # 添加实体名称
        node_data["entity_name"] = entity_name
        # 设置实体
        self._set_entity_(entity_name, node_data)
        # 添加到所有关系数据列表
        all_relationships_data.append(node_data)

    # 合并边的异步方法
    async def _merge_edges(
            self,
            src_id: str,
            tgt_id: str,
            edges_data: list[dict],
            all_relationships_data=None
    ):
        # 如果边数据列表为空，直接返回
        if not edges_data:
            return
        # 初始化已有权重列表
        already_weights = []
        # 初始化已有源ID列表
        already_source_ids = []
        # 初始化已有描述列表
        already_description = []
        # 初始化已有关键词列表
        already_keywords = []

        # 尝试获取已有关系
        relation = self._get_relation_(src_id, tgt_id)
        # 如果已有关系存在
        if relation:
            # 添加已有权重
            already_weights = [relation["weight"]]
            # 添加已有源ID
            already_source_ids = relation["source_id"]
            # 添加已有描述
            already_description = [relation["description"]]
            # 添加已有关键词
            already_keywords = relation["keywords"]

        # 计算权重总和
        weight = sum([dp["weight"] for dp in edges_data] + already_weights)
        # 合并描述，使用分隔符连接并去重
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in edges_data] + already_description))
        )
        # 合并关键词
        keywords = flat_uniq_list(edges_data, "keywords") + already_keywords
        # 合并源ID
        source_id = flat_uniq_list(edges_data, "source_id") + already_source_ids

        # 确保源节点和目标节点存在
        for need_insert_id in [src_id, tgt_id]:
            # 如果节点已存在，跳过
            if self._get_entity_(need_insert_id):
                continue
            # 创建新节点
            self._set_entity_(need_insert_id, {
                        "source_id": source_id,
                        "description": description,
                        "entity_type": 'UNKNOWN'
                    })
        # 处理关系描述摘要
        description = await self._handle_entity_relation_summary(
            f"({src_id}, {tgt_id})", description
        )
        # 构建边数据
        edge_data = dict(
            src_id=src_id,
            tgt_id=tgt_id,
            description=description,
            keywords=keywords,
            weight=weight,
            source_id=source_id
        )
        # 设置关系
        self._set_relation_(src_id, tgt_id, edge_data)
        # 如果提供了关系数据列表，添加边数据
        if all_relationships_data is not None:
            all_relationships_data.append(edge_data)

    # 处理实体或关系描述摘要的异步方法
    async def _handle_entity_relation_summary(
            self,
            entity_or_relation_name: str,
            description: str
    ) -> str:
        # 设置摘要最大令牌数
        summary_max_tokens = 512
        # 截断描述以适应最大令牌数
        use_description = truncate(description, summary_max_tokens)
        # 按分隔符分割描述
        description_list=use_description.split(GRAPH_FIELD_SEP),
        # 如果描述列表不超过12个，直接返回原描述
        if len(description_list) <= 12:
            return use_description
        # 获取摘要提示模板
        prompt_template = SUMMARIZE_DESCRIPTIONS_PROMPT
        # 构建上下文基础
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=description_list,
            language=self._language,
        )
        # 格式化提示模板
        use_prompt = prompt_template.format(**context_base)
        # 记录触发摘要
        logging.info(f"Trigger summary: {entity_or_relation_name}")
        # 使用聊天限制器异步调用LLM
        async with chat_limiter:
            summary = await trio.to_thread.run_sync(lambda: self._chat(use_prompt, [{"role": "user", "content": "Output: "}], {"temperature": 0.8}))
        # 返回摘要
        return summary
