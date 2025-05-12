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
# 导入集合模块
import collections
# 导入正则表达式模块
import re
# 导入类型注解模块
from typing import Any
# 导入数据类模块
from dataclasses import dataclass
# 导入异步处理库
import trio

# 导入提取器基类
from graphrag.general.extractor import Extractor
# 导入思维导图提取提示模板
from graphrag.general.mind_map_prompt import MIND_MAP_EXTRACTION_PROMPT
# 导入工具函数
from graphrag.utils import ErrorHandlerFn, perform_variable_replacements, chat_limiter
# 导入聊天模型基类
from rag.llm.chat_model import Base as CompletionLLM
# 导入Markdown转JSON工具
import markdown_to_json
# 导入函数式编程工具
from functools import reduce
# 导入令牌计数工具
from rag.utils import num_tokens_from_string


# 定义思维导图结果数据类
@dataclass
class MindMapResult:
    """Unipartite Mind Graph result class definition."""
    # 输出结果字典
    output: dict


# 定义思维导图提取器类
class MindMapExtractor(Extractor):
    # 输入文本键名
    _input_text_key: str
    # 思维导图提示模板
    _mind_map_prompt: str
    # 错误处理函数
    _on_error: ErrorHandlerFn

    # 初始化方法
    def __init__(
            self,
            llm_invoker: CompletionLLM,
            prompt: str | None = None,
            input_text_key: str | None = None,
            on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        # 设置LLM调用器
        self._llm = llm_invoker
        # 设置输入文本键名，默认为"input_text"
        self._input_text_key = input_text_key or "input_text"
        # 设置思维导图提示模板，默认使用预定义模板
        self._mind_map_prompt = prompt or MIND_MAP_EXTRACTION_PROMPT
        # 设置错误处理函数，默认为空函数
        self._on_error = on_error or (lambda _e, _s, _d: None)

    # 处理键名，移除星号
    def _key(self, k):
        return re.sub(r"\*+", "", k)

    # 递归构建子节点结构
    def _be_children(self, obj: dict, keyset: set):
        # 如果对象是字符串，转换为列表
        if isinstance(obj, str):
            obj = [obj]
        # 如果对象是列表，处理为子节点
        if isinstance(obj, list):
            # 将列表项添加到键集合
            keyset.update(obj)
            # 移除每个项中的星号
            obj = [re.sub(r"\*+", "", i) for i in obj]
            # 返回子节点列表，每个非空项作为一个节点
            return [{"id": i, "children": []} for i in obj if i]
        # 初始化结果数组
        arr = []
        # 遍历对象的键值对
        for k, v in obj.items():
            # 处理键名，移除星号
            k = self._key(k)
            # 如果键名有效且不在键集合中
            if k and k not in keyset:
                # 将键名添加到键集合
                keyset.add(k)
                # 添加节点到结果数组
                arr.append(
                    {
                        "id": k,
                        "children": self._be_children(v, keyset)
                    }
                )
        # 返回结果数组
        return arr

    # 调用方法，处理文本段落生成思维导图
    async def __call__(
            self, sections: list[str], prompt_variables: dict[str, Any] | None = None
    ) -> MindMapResult:
        """Call method definition."""
        # 如果未提供提示变量，初始化为空字典
        if prompt_variables is None:
            prompt_variables = {}

        # 初始化结果列表
        res = []
        # 计算令牌数量限制
        token_count = max(self._llm.max_length * 0.8, self._llm.max_length - 512)
        # 初始化文本段落列表
        texts = []
        # 初始化当前令牌计数
        cnt = 0
        # 创建异步任务组
        async with trio.open_nursery() as nursery:
            # 遍历所有文本段落
            for i in range(len(sections)):
                # 计算当前段落的令牌数
                section_cnt = num_tokens_from_string(sections[i])
                # 如果添加当前段落会超出令牌限制且已有文本
                if cnt + section_cnt >= token_count and texts:
                    # 启动异步任务处理当前文本
                    nursery.start_soon(lambda: self._process_document("".join(texts), prompt_variables, res))
                    # 重置文本列表
                    texts = []
                    # 重置令牌计数
                    cnt = 0
                # 添加当前段落到文本列表
                texts.append(sections[i])
                # 更新令牌计数
                cnt += section_cnt
            # 处理剩余文本
            if texts:
                nursery.start_soon(lambda: self._process_document("".join(texts), prompt_variables, res))
        # 如果没有结果，返回空根节点
        if not res:
            return MindMapResult(output={"id": "root", "children": []})
        # 合并所有结果
        merge_json = reduce(self._merge, res)
        # 如果有多个顶级节点
        if len(merge_json) > 1:
            # 提取所有有效键名
            keys = [re.sub(r"\*+", "", k) for k, v in merge_json.items() if isinstance(v, dict)]
            # 创建键集合
            keyset = set(i for i in keys if i)
            # 构建根节点结构
            merge_json = {
                "id": "root",
                "children": [
                    {
                        "id": self._key(k),
                        "children": self._be_children(v, keyset)
                    }
                    for k, v in merge_json.items() if isinstance(v, dict) and self._key(k)
                ]
            }
        # 如果只有一个顶级节点
        else:
            # 获取键名
            k = self._key(list(merge_json.keys())[0])
            # 构建单节点结构
            merge_json = {"id": k, "children": self._be_children(list(merge_json.items())[0][1], {k})}

        # 返回思维导图结果
        return MindMapResult(output=merge_json)

    # 合并两个字典
    def _merge(self, d1, d2):
        # 遍历第一个字典的所有键
        for k in d1:
            # 如果键在第二个字典中存在
            if k in d2:
                # 如果两个值都是字典，递归合并
                if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    self._merge(d1[k], d2[k])
                # 如果两个值都是列表，合并列表
                elif isinstance(d1[k], list) and isinstance(d2[k], list):
                    d2[k].extend(d1[k])
                # 其他情况，用第一个字典的值覆盖第二个
                else:
                    d2[k] = d1[k]
            # 如果键在第二个字典中不存在，添加该键值对
            else:
                d2[k] = d1[k]

        # 返回合并后的字典
        return d2

    # 将列表转换为键值对
    def _list_to_kv(self, data):
        # 遍历数据的所有键值对
        for key, value in data.items():
            # 如果值是字典，递归处理
            if isinstance(value, dict):
                self._list_to_kv(value)
            # 如果值是列表，转换为键值对
            elif isinstance(value, list):
                # 创建新的值字典
                new_value = {}
                # 遍历列表
                for i in range(len(value)):
                    # 如果当前项是列表且不是第一项
                    if isinstance(value[i], list) and i > 0:
                        # 使用前一项作为键，当前项的第一个元素作为值
                        new_value[value[i - 1]] = value[i][0]
                # 更新数据中的值
                data[key] = new_value
            # 其他情况，继续处理下一项
            else:
                continue
        # 返回处理后的数据
        return data

    # 将OrderedDict转换为普通字典
    def _todict(self, layer: collections.OrderedDict):
        # 初始化返回值
        to_ret = layer
        # 如果是OrderedDict，转换为普通字典
        if isinstance(layer, collections.OrderedDict):
            to_ret = dict(layer)

        try:
            # 遍历字典的所有键值对
            for key, value in to_ret.items():
                # 递归处理值
                to_ret[key] = self._todict(value)
        except AttributeError:
            # 忽略非字典类型的错误
            pass

        # 返回处理后的字典，并转换列表为键值对
        return self._list_to_kv(to_ret)

    # 处理文档，生成思维导图
    async def _process_document(
            self, text: str, prompt_variables: dict[str, str], out_res
    ) -> str:
        # 合并提示变量和输入文本
        variables = {
            **prompt_variables,
            self._input_text_key: text,
        }
        # 替换提示模板中的变量
        text = perform_variable_replacements(self._mind_map_prompt, variables=variables)
        # 设置生成配置
        gen_conf = {"temperature": 0.5}
        # 使用聊天限制器控制并发
        async with chat_limiter:
            # 异步调用聊天模型
            response = await trio.to_thread.run_sync(lambda: self._chat(text, [{"role": "user", "content": "Output:"}], gen_conf))
        # 移除Markdown代码块标记
        response = re.sub(r"```[^\n]*", "", response)
        # 记录响应到日志
        logging.debug(response)
        # 将Markdown转换为字典并记录到日志
        logging.debug(self._todict(markdown_to_json.dictify(response)))
        # 将处理结果添加到输出结果列表
        out_res.append(self._todict(markdown_to_json.dictify(response)))
