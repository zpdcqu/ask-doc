# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

# 导入类型注解模块
from typing import Any
# 导入数值计算库
import numpy as np
# 导入网络图处理库
import networkx as nx
# 导入数据类装饰器
from dataclasses import dataclass
# 导入获取最大连通分量的函数
from graphrag.general.leiden import stable_largest_connected_component
# 导入图形学习库
import graspologic as gc


# 定义节点嵌入数据类
@dataclass
class NodeEmbeddings:
    """Node embeddings class definition."""

    # 节点列表
    nodes: list[str]
    # 嵌入向量数组
    embeddings: np.ndarray


# 使用Node2Vec算法生成节点嵌入
def embed_nod2vec(
    # 输入图，可以是无向图或有向图
    graph: nx.Graph | nx.DiGraph,
    # 嵌入向量维度，默认为1536
    dimensions: int = 1536,
    # 每个节点的随机游走次数，默认为10
    num_walks: int = 10,
    # 每次随机游走的长度，默认为40
    walk_length: int = 40,
    # 上下文窗口大小，默认为2
    window_size: int = 2,
    # 训练迭代次数，默认为3
    iterations: int = 3,
    # 随机种子，默认为86
    random_seed: int = 86,
) -> NodeEmbeddings:
    """Generate node embeddings using Node2Vec."""
    # 生成嵌入向量
    lcc_tensors = gc.embed.node2vec_embed(  # type: ignore
        graph=graph,
        dimensions=dimensions,
        window_size=window_size,
        iterations=iterations,
        num_walks=num_walks,
        walk_length=walk_length,
        random_seed=random_seed,
    )
    # 返回节点嵌入对象，包含嵌入向量和对应的节点
    return NodeEmbeddings(embeddings=lcc_tensors[0], nodes=lcc_tensors[1])


# 运行节点嵌入生成流程
def run(graph: nx.Graph, args: dict[str, Any]) -> NodeEmbeddings:
    """Run method definition."""
    # 如果use_lcc参数为True（默认），则使用最大连通分量
    if args.get("use_lcc", True):
        graph = stable_largest_connected_component(graph)

    # 使用node2vec创建图嵌入
    embeddings = embed_nod2vec(
        graph=graph,
        # 从参数中获取嵌入维度，默认为1536
        dimensions=args.get("dimensions", 1536),
        # 从参数中获取随机游走次数，默认为10
        num_walks=args.get("num_walks", 10),
        # 从参数中获取随机游走长度，默认为40
        walk_length=args.get("walk_length", 40),
        # 从参数中获取窗口大小，默认为2
        window_size=args.get("window_size", 2),
        # 从参数中获取迭代次数，默认为3
        iterations=args.get("iterations", 3),
        # 从参数中获取随机种子，默认为86
        random_seed=args.get("random_seed", 86),
    )

    # 将节点和对应的嵌入向量组成对
    pairs = zip(embeddings.nodes, embeddings.embeddings.tolist(), strict=True)
    # 按节点名称排序
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    # 返回排序后的节点-嵌入对字典
    return dict(sorted_pairs)