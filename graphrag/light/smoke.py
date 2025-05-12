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

# 导入参数解析模块
import argparse
# 导入JSON处理模块
import json
# 导入应用设置模块
from api import settings
# 导入网络图处理库
import networkx as nx
# 导入日志模块
import logging
# 导入异步处理库
import trio

# 导入LLM类型枚举
from api.db import LLMType
# 导入文档服务
from api.db.services.document_service import DocumentService
# 导入知识库服务
from api.db.services.knowledgebase_service import KnowledgebaseService
# 导入LLM服务包
from api.db.services.llm_service import LLMBundle
# 导入用户服务
from api.db.services.user_service import TenantService
# 导入图更新函数
from graphrag.general.index import update_graph
# 导入图提取器
from graphrag.light.graph_extractor import GraphExtractor

# 初始化应用设置
settings.init_settings()


# 定义回调函数，用于处理进度和消息
def callback(prog=None, msg="Processing..."):
    logging.info(msg)


# 定义主异步函数
async def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加租户ID参数
    parser.add_argument(
        "-t",
        "--tenant_id",
        default=False,
        help="Tenant ID",
        action="store",
        required=True,
    )
    # 添加文档ID参数
    parser.add_argument(
        "-d",
        "--doc_id",
        default=False,
        help="Document ID",
        action="store",
        required=True,
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据ID获取文档
    e, doc = DocumentService.get_by_id(args.doc_id)
    # 如果文档不存在，抛出异常
    if not e:
        raise LookupError("Document not found.")
    # 获取文档所属的知识库ID
    kb_id = doc.kb_id

    # 从检索器获取文档分块内容
    chunks = [
        d["content_with_weight"]
        for d in settings.retrievaler.chunk_list(
            args.doc_id,
            args.tenant_id,
            [kb_id],
            max_count=6,
            fields=["content_with_weight"],
        )
    ]

    # 获取租户信息
    _, tenant = TenantService.get_by_id(args.tenant_id)
    # 创建LLM服务包
    llm_bdl = LLMBundle(args.tenant_id, LLMType.CHAT, tenant.llm_id)
    # 获取知识库信息
    _, kb = KnowledgebaseService.get_by_id(kb_id)
    # 创建嵌入服务包
    embed_bdl = LLMBundle(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)

    # 更新图并获取结果
    graph, doc_ids = await update_graph(
        GraphExtractor,
        args.tenant_id,
        kb_id,
        args.doc_id,
        chunks,
        "English",
        llm_bdl,
        embed_bdl,
        callback,
    )

    # 打印图的节点链接数据，转换为JSON格式
    print(json.dumps(nx.node_link_data(graph), ensure_ascii=False, indent=2))


# 程序入口点
if __name__ == "__main__":
    # 运行主异步函数
    trio.run(main)
