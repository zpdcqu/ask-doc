# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

# 导入日志模块
import logging
# 导入HTML处理模块
import html
# 导入类型注解模块
from typing import Any, cast
# 导入层次化Leiden社区检测算法
from graspologic.partition import hierarchical_leiden
# 导入最大连通分量工具
from graspologic.utils import largest_connected_component
# 导入网络图处理库
import networkx as nx
# 导入图空判断函数
from networkx import is_empty


# 稳定化图结构函数
def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    # 根据原图是否有向创建新图
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    # 获取所有节点及其数据
    sorted_nodes = graph.nodes(data=True)
    # 按节点ID排序
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    # 将排序后的节点添加到新图
    fixed_graph.add_nodes_from(sorted_nodes)
    # 获取所有边及其数据
    edges = list(graph.edges(data=True))

    # If the graph is undirected, we create the edges in a stable way, so we get the same results
    # for example:
    # A -> B
    # in graph theory is the same as
    # B -> A
    # in an undirected graph
    # however, this can lead to downstream issues because sometimes
    # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
    # but they base some of their logic on the order of the nodes, so the order ends up being important
    # so we sort the nodes in the edge in a stable way, so that we always get the same order
    # 如果图是无向的，以稳定方式创建边
    if not graph.is_directed():
        # 定义边排序函数
        def _sort_source_target(edge):
            source, target, edge_data = edge
            # 确保源节点ID小于目标节点ID
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        # 对所有边应用排序函数
        edges = [_sort_source_target(edge) for edge in edges]

    # 定义获取边键的函数
    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    # 按边键排序所有边
    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    # 将排序后的边添加到新图
    fixed_graph.add_edges_from(edges)
    # 返回稳定化的图
    return fixed_graph


# 规范化节点名称函数
def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """Normalize node names."""
    # 创建节点映射，将所有节点名转为大写并去除空格
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    # 使用映射重命名节点
    return nx.relabel_nodes(graph, node_mapping)


# 获取稳定的最大连通分量函数
def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    # 复制图
    graph = graph.copy()
    # 获取最大连通分量
    graph = cast(nx.Graph, largest_connected_component(graph))
    # 规范化节点名称
    graph = normalize_node_names(graph)
    # 稳定化图结构
    return _stabilize_graph(graph)


# 计算Leiden社区函数
def _compute_leiden_communities(
        graph: nx.Graph | nx.DiGraph,
        max_cluster_size: int,
        use_lcc: bool,
        seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """Return Leiden root communities."""
    # 初始化结果字典
    results: dict[int, dict[str, int]] = {}
    # 如果图为空，直接返回空结果
    if is_empty(graph):
        return results
    # 如果使用最大连通分量
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    # 使用层次化Leiden算法计算社区
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    # 处理社区映射结果
    for partition in community_mapping:
        # 确保每个层级有对应的字典
        results[partition.level] = results.get(partition.level, {})
        # 记录节点所属的社区
        results[partition.level][partition.node] = partition.cluster

    # 返回社区结果
    return results


# 运行Leiden社区检测的主函数
def run(graph: nx.Graph, args: dict[str, Any]) -> dict[int, dict[str, dict]]:
    """Run method definition."""
    # 获取最大集群大小参数，默认为12
    max_cluster_size = args.get("max_cluster_size", 12)
    # 获取是否使用最大连通分量参数，默认为True
    use_lcc = args.get("use_lcc", True)
    # 如果启用详细日志
    if args.get("verbose", False):
        # 记录运行参数
        logging.debug(
            "Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc
        )
    # 如果图中没有节点，返回空结果
    if not graph.nodes():
        return {}

    # 计算节点到社区的映射
    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    # 获取要处理的层级
    levels = args.get("levels")

    # 如果未指定层级，使用所有层级
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    # 初始化按层级组织的结果字典
    results_by_level: dict[int, dict[str, list[str]]] = {}
    # 处理每个层级
    for level in levels:
        # 初始化当前层级的结果
        result = {}
        results_by_level[level] = result
        # 处理当前层级的每个节点及其社区
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            # 如果节点不在图中，记录警告并跳过
            if node_id not in graph.nodes:
                logging.warning(f"Node {node_id} not found in the graph.")
                continue
            # 将社区ID转为字符串
            community_id = str(raw_community_id)
            # 如果社区不在结果中，初始化
            if community_id not in result:
                result[community_id] = {"weight": 0, "nodes": []}
            # 将节点添加到社区
            result[community_id]["nodes"].append(node_id)
            # 累加社区权重
            result[community_id]["weight"] += graph.nodes[node_id].get("rank", 0) * graph.nodes[node_id].get("weight", 1)
        # 获取所有社区权重
        weights = [comm["weight"] for _, comm in result.items()]
        # 如果没有权重，跳过当前层级
        if not weights:
            continue
        # 计算最大权重
        max_weight = max(weights)
        # 如果最大权重为0，跳过当前层级
        if max_weight == 0:
            continue
        # 归一化所有社区权重
        for _, comm in result.items():
            comm["weight"] /= max_weight

    # 返回按层级组织的结果
    return results_by_level


# 向图中添加社区信息的函数
def add_community_info2graph(graph: nx.Graph, nodes: list[str], community_title):
    # 遍历所有节点
    for n in nodes:
        # 如果节点没有communities属性，初始化为空列表
        if "communities" not in graph.nodes[n]:
            graph.nodes[n]["communities"] = []
        # 添加社区标题到节点的communities属性
        graph.nodes[n]["communities"].append(community_title)
        # 去除重复的社区标题
        graph.nodes[n]["communities"] = list(set(graph.nodes[n]["communities"]))
