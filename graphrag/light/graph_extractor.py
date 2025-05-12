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
# 导入提取器基类和最大提取数量常量
from graphrag.general.extractor import Extractor, ENTITY_EXTRACTION_MAX_GLEANINGS
# 导入图提示模板
from graphrag.light.graph_prompt import PROMPTS
# 导入工具函数，用于处理消息和字符串分割
from graphrag.utils import pack_user_ass_to_openai_messages, split_string_by_multi_markers, chat_limiter
# 导入聊天模型基类
from rag.llm.chat_model import Base as CompletionLLM
# 导入网络图处理库
import networkx as nx
# 导入令牌计数工具
from rag.utils import num_tokens_from_string
# 导入异步处理库
import trio


# 定义图提取结果数据类
@dataclass
class GraphExtractionResult:
    """Unipartite graph extraction result class definition."""

    # 输出图
    output: nx.Graph
    # 源文档字典
    source_docs: dict[Any, Any]


# 定义图提取器类，继承自提取器基类
class GraphExtractor(Extractor):

    # 最大提取数量
    _max_gleanings: int

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
        example_number: int = 2,
        max_gleanings: int | None = None,
    ):
        # 调用父类初始化方法
        super().__init__(llm_invoker, language, entity_types, get_entity, set_entity, get_relation, set_relation)
        """Init method definition."""
        # 设置最大提取数量，如果未提供则使用默认值
        self._max_gleanings = (
            max_gleanings
            if max_gleanings is not None
            else ENTITY_EXTRACTION_MAX_GLEANINGS
        )
        # 设置示例数量
        self._example_number = example_number
        # 根据示例数量拼接示例字符串
        examples = "\n".join(
                PROMPTS["entity_extraction_examples"][: int(self._example_number)]
            )

        # 创建示例上下文基础字典
        example_context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self._entity_types),
            language=self._language,
        )
        # 使用上下文格式化示例
        examples = examples.format(**example_context_base)

        # 设置实体提取提示模板
        self._entity_extract_prompt = PROMPTS["entity_extraction"]
        # 创建上下文基础字典，包含示例
        self._context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self._entity_types),
            examples=examples,
            language=self._language,
        )

        # 设置继续提取提示
        self._continue_prompt = PROMPTS["entiti_continue_extraction"]
        # 设置循环判断提示
        self._if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

        # 计算剩余令牌数量，用于控制输入文本长度
        self._left_token_count = llm_invoker.max_length - num_tokens_from_string(
            self._entity_extract_prompt.format(
                **self._context_base, input_text="{input_text}"
            ).format(**self._context_base, input_text="")
        )
        # 确保剩余令牌数量至少为最大长度的60%
        self._left_token_count = max(llm_invoker.max_length * 0.6, self._left_token_count)

    # 处理单个内容块的异步方法
    async def _process_single_content(self, chunk_key_dp: tuple[str, str], chunk_seq: int, num_chunks: int, out_results):
        # 初始化令牌计数
        token_count = 0
        # 获取块键和内容
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        # 格式化提示模板
        hint_prompt = self._entity_extract_prompt.format(
            **self._context_base, input_text="{input_text}"
        ).format(**self._context_base, input_text=content)

        # 设置生成配置，温度为0.8
        gen_conf = {"temperature": 0.8}
        # 使用聊天限制器控制并发
        async with chat_limiter:
            # 异步调用聊天模型获取初始结果
            final_result = await trio.to_thread.run_sync(lambda: self._chat(hint_prompt, [{"role": "user", "content": "Output:"}], gen_conf))
        # 累加令牌计数
        token_count += num_tokens_from_string(hint_prompt + final_result)
        # 打包用户和助手消息，准备继续对话
        history = pack_user_ass_to_openai_messages("Output:", final_result, self._continue_prompt)
        # 循环提取更多实体，最多提取_max_gleanings次
        for now_glean_index in range(self._max_gleanings):
            # 使用聊天限制器控制并发
            async with chat_limiter:
                # 异步调用聊天模型获取额外提取结果
                glean_result = await trio.to_thread.run_sync(lambda: self._chat(hint_prompt, history, gen_conf))
            # 将新结果添加到历史记录中
            history.extend([{"role": "assistant", "content": glean_result}, {"role": "user", "content": self._continue_prompt}])
            # 累加令牌计数
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + hint_prompt + self._continue_prompt)
            # 将新结果添加到最终结果中
            final_result += glean_result
            # 如果达到最大提取次数，则跳出循环
            if now_glean_index == self._max_gleanings - 1:
                break

            # 使用聊天限制器控制并发
            async with chat_limiter:
                # 异步调用聊天模型判断是否需要继续提取
                if_loop_result = await trio.to_thread.run_sync(lambda: self._chat(self._if_loop_prompt, history, gen_conf))
            # 累加令牌计数
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + if_loop_result + self._if_loop_prompt)
            # 处理判断结果，去除多余字符并转为小写
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            # 如果判断结果不是"yes"，则跳出循环
            if if_loop_result != "yes":
                break

        # 使用分隔符分割最终结果，获取记录列表
        records = split_string_by_multi_markers(
            final_result,
            [self._context_base["record_delimiter"], self._context_base["completion_delimiter"]],
        )
        # 初始化处理后的记录列表
        rcds = []
        # 使用正则表达式提取每条记录中的括号内容
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            rcds.append(record.group(1))
        # 更新记录列表
        records = rcds
        # 从记录中提取实体和关系
        maybe_nodes, maybe_edges = self._entities_and_relations(chunk_key, records, self._context_base["tuple_delimiter"])
        # 将结果添加到输出结果列表中
        out_results.append((maybe_nodes, maybe_edges, token_count))
        # 如果有回调函数，则调用以报告进度
        if self.callback:
            self.callback(0.5+0.1*len(out_results)/num_chunks, msg = f"Entities extraction of chunk {chunk_seq} {len(out_results)}/{num_chunks} done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {token_count} tokens.")
