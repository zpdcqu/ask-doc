# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

# 导入正则表达式模块
import re
# 导入类型注解和可调用对象
from typing import Any, Callable
# 导入数据类装饰器
from dataclasses import dataclass
# 导入tiktoken用于令牌计数
import tiktoken
# 导入异步处理库
import trio

# 导入提取器基类和相关常量
from graphrag.general.extractor import Extractor, ENTITY_EXTRACTION_MAX_GLEANINGS, DEFAULT_ENTITY_TYPES
# 导入图提取相关提示模板
from graphrag.general.graph_prompt import GRAPH_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT
# 导入工具函数
from graphrag.utils import ErrorHandlerFn, perform_variable_replacements, chat_limiter
# 导入聊天模型基类
from rag.llm.chat_model import Base as CompletionLLM
# 导入网络图处理库
import networkx as nx
# 导入令牌计数工具
from rag.utils import num_tokens_from_string

# 定义默认元组分隔符
DEFAULT_TUPLE_DELIMITER = "<|>"
# 定义默认记录分隔符
DEFAULT_RECORD_DELIMITER = "##"
# 定义默认完成分隔符
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


# 定义图提取结果数据类
@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    # 输出图结构
    output: nx.Graph
    # 源文档字典
    source_docs: dict[Any, Any]


# 定义图提取器类
class GraphExtractor(Extractor):
    """Unipartite graph extractor class definition."""

    # 是否合并描述
    _join_descriptions: bool
    # 元组分隔符键名
    _tuple_delimiter_key: str
    # 记录分隔符键名
    _record_delimiter_key: str
    # 实体类型键名
    _entity_types_key: str
    # 输入文本键名
    _input_text_key: str
    # 完成分隔符键名
    _completion_delimiter_key: str
    # 实体名称键名
    _entity_name_key: str
    # 输入描述键名
    _input_descriptions_key: str
    # 提取提示模板
    _extraction_prompt: str
    # 摘要提示模板
    _summarization_prompt: str
    # 循环参数字典
    _loop_args: dict[str, Any]
    # 最大提取次数
    _max_gleanings: int
    # 错误处理函数
    _on_error: ErrorHandlerFn

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
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        input_text_key: str | None = None,
        entity_types_key: str | None = None,
        completion_delimiter_key: str | None = None,
        join_descriptions=True,
        max_gleanings: int | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        # 调用父类初始化方法
        super().__init__(llm_invoker, language, entity_types, get_entity, set_entity, get_relation, set_relation)
        """Init method definition."""
        # TODO: streamline construction
        # 设置LLM调用器
        self._llm = llm_invoker
        # 设置是否合并描述
        self._join_descriptions = join_descriptions
        # 设置输入文本键名，默认为"input_text"
        self._input_text_key = input_text_key or "input_text"
        # 设置元组分隔符键名，默认为"tuple_delimiter"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        # 设置记录分隔符键名，默认为"record_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        # 设置完成分隔符键名，默认为"completion_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )
        # 设置实体类型键名，默认为"entity_types"
        self._entity_types_key = entity_types_key or "entity_types"
        # 设置提取提示模板
        self._extraction_prompt = GRAPH_EXTRACTION_PROMPT
        # 设置最大提取次数，默认使用预定义常量
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        # 设置错误处理函数，默认为空函数
        self._on_error = on_error or (lambda _e, _s, _d: None)
        # 计算提示模板的令牌数
        self.prompt_token_count = num_tokens_from_string(self._extraction_prompt)

        # 构建循环参数
        # 获取cl100k_base编码器
        encoding = tiktoken.get_encoding("cl100k_base")
        # 编码"YES"
        yes = encoding.encode("YES")
        # 编码"NO"
        no = encoding.encode("NO")
        # 设置循环参数，包括logit偏置和最大令牌数
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}

        # 将默认值设置到提示变量中
        self._prompt_variables = {
            "entity_types": entity_types,
            self._tuple_delimiter_key: DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(DEFAULT_ENTITY_TYPES),
        }

    # 处理单个内容块的异步方法
    async def _process_single_content(self, chunk_key_dp: tuple[str, str], chunk_seq: int, num_chunks: int, out_results):
        # 初始化令牌计数
        token_count = 0
        # 获取块键
        chunk_key = chunk_key_dp[0]
        # 获取内容
        content = chunk_key_dp[1]
        # 设置变量字典，包含提示变量和输入文本
        variables = {
            **self._prompt_variables,
            self._input_text_key: content,
        }
        # 设置生成配置，温度为0.3
        gen_conf = {"temperature": 0.3}
        # 替换提示模板中的变量
        hint_prompt = perform_variable_replacements(self._extraction_prompt, variables=variables)
        # 使用聊天限制器异步调用LLM
        async with chat_limiter:
            response = await trio.to_thread.run_sync(lambda: self._chat(hint_prompt, [{"role": "user", "content": "Output:"}], gen_conf))
        # 累加令牌计数
        token_count += num_tokens_from_string(hint_prompt + response)

        # 初始化结果字符串
        results = response or ""
        # 初始化对话历史
        history = [{"role": "system", "content": hint_prompt}, {"role": "user", "content": response}]

        # 重复提取以确保最大化实体数量
        for i in range(self._max_gleanings):
            # 替换继续提示模板中的变量
            text = perform_variable_replacements(CONTINUE_PROMPT, history=history, variables=variables)
            # 添加用户消息到历史
            history.append({"role": "user", "content": text})
            # 使用聊天限制器异步调用LLM
            async with chat_limiter:
                response = await trio.to_thread.run_sync(lambda: self._chat("", history, gen_conf))
            # 累加令牌计数
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + response)
            # 添加响应到结果
            results += response or ""

            # 如果这是最后一次提取，不需要更新继续标志
            if i >= self._max_gleanings - 1:
                break
            # 添加助手响应到历史
            history.append({"role": "assistant", "content": response})
            # 添加循环提示到历史
            history.append({"role": "user", "content": LOOP_PROMPT})
            # 使用聊天限制器异步调用LLM判断是否继续
            async with chat_limiter:
                continuation = await trio.to_thread.run_sync(lambda: self._chat("", history, {"temperature": 0.8}))
            # 累加令牌计数
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + response)
            # 如果不继续，跳出循环
            if continuation != "YES":
                break
        # 获取记录分隔符
        record_delimiter = variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER)
        # 获取元组分隔符
        tuple_delimiter = variables.get(self._tuple_delimiter_key, DEFAULT_TUPLE_DELIMITER)
        # 分割结果为记录，并移除括号
        records = [re.sub(r"^\(|\)$", "", r.strip()) for r in results.split(record_delimiter)]
        # 过滤空记录
        records = [r for r in records if r.strip()]
        # 提取实体和关系
        maybe_nodes, maybe_edges = self._entities_and_relations(chunk_key, records, tuple_delimiter)
        # 将结果添加到输出列表
        out_results.append((maybe_nodes, maybe_edges, token_count))
        # 如果设置了回调函数，调用回调
        if self.callback:
            self.callback(0.5+0.1*len(out_results)/num_chunks, msg = f"Entities extraction of chunk {chunk_seq} {len(out_results)}/{num_chunks} done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {token_count} tokens.")
