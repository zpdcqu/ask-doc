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
# 导入所需的库
import logging
# 导入日志模块
import itertools
# 导入迭代工具模块
import re
# 导入正则表达式模块
import time
# 导入时间模块
from dataclasses import dataclass
# 从dataclasses导入dataclass装饰器
from typing import Any, Callable
# 从typing导入Any和Callable类型

# 导入网络图处理库
import networkx as nx
# 导入异步处理库
import trio

# 导入自定义提取器基类
from graphrag.general.extractor import Extractor
# 导入英文检测函数
from rag.nlp import is_english
# 导入编辑距离计算库
import editdistance
# 导入实体解析提示模板
from graphrag.entity_resolution_prompt import ENTITY_RESOLUTION_PROMPT
# 导入LLM聊天模型基类
from rag.llm.chat_model import Base as CompletionLLM
# 导入工具函数
from graphrag.utils import perform_variable_replacements, chat_limiter

# 默认记录分隔符
DEFAULT_RECORD_DELIMITER = "##"
# 默认实体索引分隔符
DEFAULT_ENTITY_INDEX_DELIMITER = "<|>"
# 默认解析结果分隔符
DEFAULT_RESOLUTION_RESULT_DELIMITER = "&&"


# 定义实体解析结果数据类
@dataclass
class EntityResolutionResult:
    """Entity resolution result class definition."""
    # 处理后的图
    graph: nx.Graph
    # 被移除的实体列表
    removed_entities: list


# 定义实体解析类，继承自Extractor
class EntityResolution(Extractor):
    """Entity resolution class definition."""

    # 解析提示模板
    _resolution_prompt: str
    # 输出格式化提示模板
    _output_formatter_prompt: str
    # 记录分隔符键名
    _record_delimiter_key: str
    # 实体索引分隔符键名
    _entity_index_delimiter_key: str
    # 解析结果分隔符键名
    _resolution_result_delimiter_key: str

    # 初始化方法
    def __init__(
            self,
            llm_invoker: CompletionLLM,
            get_entity: Callable | None = None,
            set_entity: Callable | None = None,
            get_relation: Callable | None = None,
            set_relation: Callable | None = None
    ):
        # 调用父类初始化方法
        super().__init__(llm_invoker, get_entity=get_entity, set_entity=set_entity, get_relation=get_relation, set_relation=set_relation)
        """Init method definition."""
        # 设置LLM调用器
        self._llm = llm_invoker
        # 设置解析提示模板
        self._resolution_prompt = ENTITY_RESOLUTION_PROMPT
        # 设置记录分隔符键名
        self._record_delimiter_key = "record_delimiter"
        # 设置实体索引分隔符键名
        self._entity_index_dilimiter_key = "entity_index_delimiter"
        # 设置解析结果分隔符键名
        self._resolution_result_delimiter_key = "resolution_result_delimiter"
        # 设置输入文本键名
        self._input_text_key = "input_text"

    # 异步调用方法，执行实体解析
    async def __call__(self, graph: nx.Graph, prompt_variables: dict[str, Any] | None = None, callback: Callable | None = None) -> EntityResolutionResult:
        """Call method definition."""
        # 如果未提供提示变量，则初始化为空字典
        if prompt_variables is None:
            prompt_variables = {}

        # 将默认值注入提示变量
        self.prompt_variables = {
            **prompt_variables,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
                                        or DEFAULT_RECORD_DELIMITER,
            self._entity_index_dilimiter_key: prompt_variables.get(self._entity_index_dilimiter_key)
                                              or DEFAULT_ENTITY_INDEX_DELIMITER,
            self._resolution_result_delimiter_key: prompt_variables.get(self._resolution_result_delimiter_key)
                                                   or DEFAULT_RESOLUTION_RESULT_DELIMITER,
        }

        # 获取图中的所有节点
        nodes = graph.nodes
        # 获取所有实体类型
        entity_types = list(set(graph.nodes[node].get('entity_type', '-') for node in nodes))
        # 按实体类型对节点进行分组
        node_clusters = {entity_type: [] for entity_type in entity_types}

        # 将每个节点添加到对应的实体类型组中
        for node in nodes:
            node_clusters[graph.nodes[node].get('entity_type', '-')].append(node)

        # 初始化候选解析字典，按实体类型分组
        candidate_resolution = {entity_type: [] for entity_type in entity_types}
        # 对每种实体类型，生成可能相同的实体对
        for k, v in node_clusters.items():
            candidate_resolution[k] = [(a, b) for a, b in itertools.combinations(v, 2) if self.is_similarity(a, b)]
        # 计算候选对的总数
        num_candidates = sum([len(candidates) for _, candidates in candidate_resolution.items()])
        # 通过回调函数报告候选对数量
        callback(msg=f"Identified {num_candidates} candidate pairs")

        # 初始化解析结果集
        resolution_result = set()
        # 使用trio创建异步任务组
        async with trio.open_nursery() as nursery:
            # 对每种实体类型的候选对进行解析
            for candidate_resolution_i in candidate_resolution.items():
                # 如果没有候选对，则跳过
                if not candidate_resolution_i[1]:
                    continue
                # 启动异步任务处理候选对
                nursery.start_soon(lambda: self._resolve_candidate(candidate_resolution_i, resolution_result))
        # 通过回调函数报告解析结果
        callback(msg=f"Resolved {num_candidates} candidate pairs, {len(resolution_result)} of them are selected to merge.")

        # 创建连接图
        connect_graph = nx.Graph()
        # 初始化被移除的实体列表
        removed_entities = []
        # 将解析结果添加为连接图的边
        connect_graph.add_edges_from(resolution_result)
        # 初始化所有实体数据列表
        all_entities_data = []
        # 初始化所有关系数据列表
        all_relationships_data = []
        # 初始化所有要移除的节点列表
        all_remove_nodes = []

        # 使用trio创建异步任务组
        async with trio.open_nursery() as nursery:
            # 对连接图中的每个连通分量进行处理
            for sub_connect_graph in nx.connected_components(connect_graph):
                # 获取子图
                sub_connect_graph = connect_graph.subgraph(sub_connect_graph)
                # 获取要移除的节点列表
                remove_nodes = list(sub_connect_graph.nodes)
                # 保留一个节点，其余节点将被移除
                keep_node = remove_nodes.pop()
                # 将要移除的节点添加到列表中
                all_remove_nodes.append(remove_nodes)
                # 启动异步任务合并节点
                nursery.start_soon(lambda: self._merge_nodes(keep_node, self._get_entity_(remove_nodes), all_entities_data))
                # 处理每个要移除的节点
                for remove_node in remove_nodes:
                    # 将节点添加到被移除实体列表
                    removed_entities.append(remove_node)
                    # 获取要移除节点的所有邻居
                    remove_node_neighbors = graph[remove_node]
                    remove_node_neighbors = list(remove_node_neighbors)
                    # 处理每个邻居节点
                    for remove_node_neighbor in remove_node_neighbors:
                        # 获取节点间的关系
                        rel = self._get_relation_(remove_node, remove_node_neighbor)
                        # 如果存在边，则移除
                        if graph.has_edge(remove_node, remove_node_neighbor):
                            graph.remove_edge(remove_node, remove_node_neighbor)
                        # 如果邻居就是保留节点，则移除它们之间的边并继续
                        if remove_node_neighbor == keep_node:
                            if graph.has_edge(keep_node, remove_node):
                                graph.remove_edge(keep_node, remove_node)
                            continue
                        # 如果没有关系，则继续
                        if not rel:
                            continue
                        # 如果保留节点与邻居已有边，则合并边
                        if graph.has_edge(keep_node, remove_node_neighbor):
                            nursery.start_soon(lambda: self._merge_edges(keep_node, remove_node_neighbor, [rel], all_relationships_data))
                        # 否则，添加新边
                        else:
                            # 对节点对进行排序
                            pair = sorted([keep_node, remove_node_neighbor])
                            # 添加边到图中
                            graph.add_edge(pair[0], pair[1], weight=rel['weight'])
                            # 设置关系属性
                            self._set_relation_(pair[0], pair[1],
                                            dict(
                                                    src_id=pair[0],
                                                    tgt_id=pair[1],
                                                    weight=rel['weight'],
                                                    description=rel['description'],
                                                    keywords=[],
                                                    source_id=rel.get("source_id", ""),
                                                    metadata={"created_at": time.time()}
                                            ))
                    # 从图中移除节点
                    graph.remove_node(remove_node)

        # 返回实体解析结果
        return EntityResolutionResult(
            graph=graph,
            removed_entities=removed_entities
        )

    # 异步方法，解析候选实体对
    async def _resolve_candidate(self, candidate_resolution_i, resolution_result):
        # 设置生成配置，温度为0.5
        gen_conf = {"temperature": 0.5}
        # 构建提示文本列表
        pair_txt = [
            f'When determining whether two {candidate_resolution_i[0]}s are the same, you should only focus on critical properties and overlook noisy factors.\n']
        # 为每个候选对添加问题
        for index, candidate in enumerate(candidate_resolution_i[1]):
            pair_txt.append(
                f'Question {index + 1}: name of{candidate_resolution_i[0]} A is {candidate[0]} ,name of{candidate_resolution_i[0]} B is {candidate[1]}')
        # 根据问题数量设置提示文本
        sent = 'question above' if len(pair_txt) == 1 else f'above {len(pair_txt)} questions'
        # 添加回答格式说明
        pair_txt.append(
            f'\nUse domain knowledge of {candidate_resolution_i[0]}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {candidate_resolution_i[0]} A and {candidate_resolution_i[0]} B are the same {candidate_resolution_i[0]}./No, {candidate_resolution_i[0]} A and {candidate_resolution_i[0]} B are different {candidate_resolution_i[0]}s. For Question i+1, (repeat the above procedures)')
        # 将提示文本列表合并为字符串
        pair_prompt = '\n'.join(pair_txt)
        # 设置变量字典
        variables = {
            **self.prompt_variables,
            self._input_text_key: pair_prompt
        }
        # 执行变量替换，生成最终提示文本
        text = perform_variable_replacements(self._resolution_prompt, variables=variables)
        # 记录日志
        logging.info(f"Created resolution prompt {len(text)} bytes for {len(candidate_resolution_i[1])} entity pairs of type {candidate_resolution_i[0]}")
        # 使用聊天限制器控制并发
        async with chat_limiter:
            # 异步调用LLM
            response = await trio.to_thread.run_sync(lambda: self._chat(text, [{"role": "user", "content": "Output:"}], gen_conf))
        # 记录调试日志
        logging.debug(f"_resolve_candidate chat prompt: {text}\nchat response: {response}")
        # 处理LLM响应结果
        result = self._process_results(len(candidate_resolution_i[1]), response,
                                       self.prompt_variables.get(self._record_delimiter_key,
                                                            DEFAULT_RECORD_DELIMITER),
                                       self.prompt_variables.get(self._entity_index_dilimiter_key,
                                                            DEFAULT_ENTITY_INDEX_DELIMITER),
                                       self.prompt_variables.get(self._resolution_result_delimiter_key,
                                                            DEFAULT_RESOLUTION_RESULT_DELIMITER))
        # 将结果添加到解析结果集
        for result_i in result:
            resolution_result.add(candidate_resolution_i[1][result_i[0] - 1])

    # 处理LLM响应结果的方法
    def _process_results(
            self,
            records_length: int,
            results: str,
            record_delimiter: str,
            entity_index_delimiter: str,
            resolution_result_delimiter: str
    ) -> list:
        # 初始化结果列表
        ans_list = []
        # 按记录分隔符分割结果
        records = [r.strip() for r in results.split(record_delimiter)]
        # 处理每条记录
        for record in records:
            # 使用正则表达式匹配实体索引
            pattern_int = f"{re.escape(entity_index_delimiter)}(\d+){re.escape(entity_index_delimiter)}"
            match_int = re.search(pattern_int, record)
            # 提取索引值，默认为0
            res_int = int(str(match_int.group(1) if match_int else '0'))
            # 如果索引超出范围，则跳过
            if res_int > records_length:
                continue

            # 使用正则表达式匹配解析结果
            pattern_bool = f"{re.escape(resolution_result_delimiter)}([a-zA-Z]+){re.escape(resolution_result_delimiter)}"
            match_bool = re.search(pattern_bool, record)
            # 提取结果值，默认为空字符串
            res_bool = str(match_bool.group(1) if match_bool else '')

            # 如果索引和结果都有效
            if res_int and res_bool:
                # 如果结果为"yes"，则添加到结果列表
                if res_bool.lower() == 'yes':
                    ans_list.append((res_int, "yes"))

        # 返回结果列表
        return ans_list

    # 判断两个实体是否相似的方法
    def is_similarity(self, a, b):
        # 如果两个实体都是英文
        if is_english(a) and is_english(b):
            # 计算编辑距离，如果小于等于较短实体长度的一半，则认为相似
            if editdistance.eval(a, b) <= min(len(a), len(b)) // 2:
                return True

        # 如果两个实体有共同字符，则认为相似
        if len(set(a) & set(b)) > 0:
            return True

        # 默认返回不相似
        return False
