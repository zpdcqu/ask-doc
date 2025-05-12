# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

# 导入日志模块
import logging
# 导入JSON处理模块
import json
# 导入正则表达式模块
import re
# 导入类型注解
from typing import Callable
# 导入数据类装饰器
from dataclasses import dataclass
# 导入网络图处理库
import networkx as nx
# 导入数据分析库
import pandas as pd
# 导入社区检测模块
from graphrag.general import leiden
# 导入社区报告提示模板
from graphrag.general.community_report_prompt import COMMUNITY_REPORT_PROMPT
# 导入提取器基类
from graphrag.general.extractor import Extractor
# 导入添加社区信息到图的函数
from graphrag.general.leiden import add_community_info2graph
# 导入聊天模型基类
from rag.llm.chat_model import Base as CompletionLLM
# 导入工具函数
from graphrag.utils import perform_variable_replacements, dict_has_keys_with_types, chat_limiter
# 导入令牌计数工具
from rag.utils import num_tokens_from_string
# 导入异步处理库
import trio


# 定义社区报告结果数据类
@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    # 文本输出列表
    output: list[str]
    # 结构化输出列表
    structured_output: list[dict]


# 定义社区报告提取器类
class CommunityReportsExtractor(Extractor):
    """Community reports extractor class definition."""

    # 提取提示模板
    _extraction_prompt: str
    # 输出格式化提示模板
    _output_formatter_prompt: str
    # 最大报告长度
    _max_report_length: int

    # 初始化方法
    def __init__(
            self,
            llm_invoker: CompletionLLM,
            get_entity: Callable | None = None,
            set_entity: Callable | None = None,
            get_relation: Callable | None = None,
            set_relation: Callable | None = None,
            max_report_length: int | None = None,
    ):
        # 调用父类初始化方法
        super().__init__(llm_invoker, get_entity=get_entity, set_entity=set_entity, get_relation=get_relation, set_relation=set_relation)
        """Init method definition."""
        # 设置LLM调用器
        self._llm = llm_invoker
        # 设置提取提示模板
        self._extraction_prompt = COMMUNITY_REPORT_PROMPT
        # 设置最大报告长度，默认为1500
        self._max_report_length = max_report_length or 1500

    # 异步调用方法，处理图并生成社区报告
    async def __call__(self, graph: nx.Graph, callback: Callable | None = None):
        # 为图中的每个节点添加度数作为排名属性
        for node_degree in graph.degree:
            graph.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])

        # 运行社区检测算法
        communities: dict[str, dict[str, list]] = leiden.run(graph, {})
        # 计算社区总数
        total = sum([len(comm.items()) for _, comm in communities.items()])
        # 初始化结果字符串列表
        res_str = []
        # 初始化结果字典列表
        res_dict = []
        # 初始化已处理社区数和令牌计数
        over, token_count = 0, 0
        # 定义提取社区报告的异步函数
        async def extract_community_report(community):
            # 声明使用外部变量
            nonlocal res_str, res_dict, over, token_count
            # 获取社区ID和实体信息
            cm_id, ents = community
            # 获取社区权重
            weight = ents["weight"]
            # 获取社区节点
            ents = ents["nodes"]
            # 获取实体数据并转换为DataFrame
            ent_df = pd.DataFrame(self._get_entity_(ents)).dropna()
            # 如果实体数据为空或没有实体名称列，则返回
            if ent_df.empty or "entity_name" not in ent_df.columns:
                return
            # 添加实体列并删除实体名称列
            ent_df["entity"] = ent_df["entity_name"]
            del ent_df["entity_name"]
            # 获取关系数据并转换为DataFrame
            rela_df = pd.DataFrame(self._get_relation_(list(ent_df["entity"]), list(ent_df["entity"]), 10000))
            # 如果关系数据为空，则返回
            if rela_df.empty:
                return
            # 重命名源和目标列
            rela_df["source"] = rela_df["src_id"]
            rela_df["target"] = rela_df["tgt_id"]
            # 删除原始源和目标ID列
            del rela_df["src_id"]
            del rela_df["tgt_id"]

            # 准备提示变量
            prompt_variables = {
                "entity_df": ent_df.to_csv(index_label="id"),
                "relation_df": rela_df.to_csv(index_label="id")
            }
            # 替换提示模板中的变量
            text = perform_variable_replacements(self._extraction_prompt, variables=prompt_variables)
            # 设置生成配置
            gen_conf = {"temperature": 0.3}
            # 使用聊天限制器异步调用LLM
            async with chat_limiter:
                response = await trio.to_thread.run_sync(lambda: self._chat(text, [{"role": "user", "content": "Output:"}], gen_conf))
            # 累加令牌计数
            token_count += num_tokens_from_string(text + response)
            # 清理响应中的非JSON部分（开头）
            response = re.sub(r"^[^\{]*", "", response)
            # 清理响应中的非JSON部分（结尾）
            response = re.sub(r"[^\}]*$", "", response)
            # 替换双大括号为单大括号（开始）
            response = re.sub(r"\{\{", "{", response)
            # 替换双大括号为单大括号（结束）
            response = re.sub(r"\}\}", "}", response)
            # 记录调试信息
            logging.debug(response)
            # 尝试解析JSON响应
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                # 记录JSON解析错误
                logging.error(f"Failed to parse JSON response: {e}")
                logging.error(f"Response content: {response}")
                return
            # 验证响应是否包含所需的键和类型
            if not dict_has_keys_with_types(response, [
                        ("title", str),
                        ("summary", str),
                        ("findings", list),
                        ("rating", float),
                        ("rating_explanation", str),
                    ]):
                return
            # 添加权重到响应
            response["weight"] = weight
            # 添加实体到响应
            response["entities"] = ents
            # 将社区信息添加到图中
            add_community_info2graph(graph, ents, response["title"])
            # 添加文本输出到结果
            res_str.append(self._get_text_output(response))
            # 添加字典输出到结果
            res_dict.append(response)
            # 增加已处理社区计数
            over += 1
            # 如果有回调函数，则调用
            if callback:
                callback(msg=f"Communities: {over}/{total}, used tokens: {token_count}")

        # 记录开始时间
        st = trio.current_time()
        # 使用异步任务组处理所有社区
        async with trio.open_nursery() as nursery:
            for level, comm in communities.items():
                # 记录每个级别的社区数量
                logging.info(f"Level {level}: Community: {len(comm.keys())}")
                for community in comm.items():
                    # 为每个社区启动异步任务
                    nursery.start_soon(lambda: extract_community_report(community))
        # 如果有回调函数，则调用以报告完成情况
        if callback:
            callback(msg=f"Community reports done in {trio.current_time() - st:.2f}s, used tokens: {token_count}")

        # 返回社区报告结果
        return CommunityReportsResult(
            structured_output=res_dict,
            output=res_str,
        )

    # 获取文本输出的方法
    def _get_text_output(self, parsed_output: dict) -> str:
        # 获取标题，默认为"Report"
        title = parsed_output.get("title", "Report")
        # 获取摘要，默认为空字符串
        summary = parsed_output.get("summary", "")
        # 获取发现列表，默认为空列表
        findings = parsed_output.get("findings", [])

        # 定义获取发现摘要的内部函数
        def finding_summary(finding: dict):
            # 如果发现是字符串，则直接返回
            if isinstance(finding, str):
                return finding
            # 否则返回摘要字段
            return finding.get("summary")

        # 定义获取发现解释的内部函数
        def finding_explanation(finding: dict):
            # 如果发现是字符串，则返回空字符串
            if isinstance(finding, str):
                return ""
            # 否则返回解释字段
            return finding.get("explanation")

        # 生成报告部分，包括每个发现的摘要和解释
        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        # 返回格式化的完整报告文本
        return f"# {title}\n\n{summary}\n\n{report_sections}"
