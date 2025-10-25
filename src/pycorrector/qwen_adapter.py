"""千问大模型适配器 - 使用 LangChain + diff 算法精确定位"""

import logging
import difflib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================== Pydantic 模型定义 ====================


class ErrorDetail(BaseModel):
    """错误详情（大模型返回）"""

    original_phrase: str = Field(description="错误的词/短语")
    corrected_phrase: str = Field(description="正确的词/短语")
    error_type: str = Field(description="错误类型：typo/semantic/grammar")
    explanation: str = Field(description="错误原因")


class QwenCorrectionOutput(BaseModel):
    """千问纠错输出结构"""

    corrected_text: str = Field(description="完整的修正后文本")
    error_details: List[ErrorDetail] = Field(description="错误详情列表")


# ==================== 适配器类 ====================


class QwenAdapter:
    """
    千问大模型适配器

    特点：
    1. 使用 LangChain 简化调用
    2. 结构化输出（Pydantic）
    3. diff 算法精确定位（100%准确）
    4. 与现有 BERT/GPT 适配器接口兼容
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-turbo",
        max_workers: int = 5,
    ):
        """
        初始化千问适配器

        Args:
            api_key: 阿里云 API Key，如果为 None 则从环境变量读取
            model: 使用的模型，默认 qwen-turbo（性价比最高）
            max_workers: 最大并发数，默认 5
        """
        self.model = model
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 从环境变量读取 API Key
        import os

        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")

        if not self.api_key:
            raise ValueError(
                "Qwen API Key 未配置！请通过以下方式之一设置：\n"
                "1. 环境变量: export DASHSCOPE_API_KEY='your-api-key'\n"
                "2. 传入参数: QwenAdapter(api_key='your-api-key')"
            )

        # 设置环境变量（dashscope SDK 需要）
        os.environ["DASHSCOPE_API_KEY"] = self.api_key

        # 延迟导入 LangChain（避免影响其他模块）
        try:
            from langchain_community.chat_models import ChatTongyi
            from langchain.output_parsers import PydanticOutputParser
            from langchain.prompts import PromptTemplate

            # 初始化千问LLM
            self.llm = ChatTongyi(
                model=model,  # 参数名是 model 不是 model_name
                temperature=0.1,  # 低温度保证稳定输出
                top_p=0.8,
            )

            # 初始化输出解析器
            self.output_parser = PydanticOutputParser(
                pydantic_object=QwenCorrectionOutput
            )

            # 构建 Prompt 模板
            self.prompt_template = PromptTemplate(
                template=self._get_prompt_template(),
                input_variables=["text"],
                partial_variables={
                    "format_instructions": self.output_parser.get_format_instructions()
                },
            )

            logger.info(f"千问适配器初始化成功，模型: {model}")

        except Exception as e:
            logger.error(f"千问适配器初始化失败: {e}")
            raise

    def _get_prompt_template(self) -> str:
        """获取 Prompt 模板"""
        return """你是一个专业的中文文本纠错助手。请仔细检查以下文本，找出所有错误（包括错别字、词语搭配不当、语义问题、语法错误等）。

原文：
{text}

要求：
1. 返回完整的修正后文本（corrected_text）
2. 列出所有发现的错误（error_details）
3. original_phrase 要写完整的错误内容（不只是单个字）
4. error_type 可选值：typo（错别字）、semantic（语义问题）、grammar（语法错误）、punctuation（标点符号）
5. explanation 简短说明错误原因

{format_instructions}

重要规则：
- 只修改明确有错误的地方，不要随意改动
- 不要添加或删除标点符号（除非原文标点有明确错误）
- 不要改变原文的格式和结构
- 如果没有错误，corrected_text 与原文完全相同，error_details 为空数组
- original_phrase 和 corrected_phrase 必须准确对应"""

    def _calculate_precise_positions(
        self, source: str, target: str, error_details: List[ErrorDetail]
    ) -> List[Dict[str, Any]]:
        """
        使用大模型的 error_details + diff 算法精确计算位置

        Args:
            source: 原始文本
            target: 修正后的文本
            error_details: 大模型返回的错误详情

        Returns:
            带有精确位置的错误列表
        """
        # 1. 用 diff 算法找出所有差异（精确位置）
        matcher = difflib.SequenceMatcher(None, source, target)
        diff_changes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":  # 只保留有变化的部分
                diff_changes.append(
                    {
                        "tag": tag,
                        "position": i1,
                        "original": source[i1:i2],
                        "corrected": target[j1:j2] if tag != "delete" else "",
                    }
                )

        # 2. 将 diff 结果和大模型的 error_details 进行匹配
        errors = []
        used_details = set()

        for change in diff_changes:
            # 尝试匹配大模型的错误详情
            matched_detail = None

            for idx, detail in enumerate(error_details):
                if idx in used_details:
                    continue

                # 增强匹配策略：考虑多种情况
                original_text = change["original"]
                detail_phrase = detail.original_phrase

                # 策略1：直接文本重叠
                if original_text and detail_phrase:
                    if original_text in detail_phrase or detail_phrase in original_text:
                        matched_detail = detail
                        used_details.add(idx)
                        break

                # 策略2：对于插入/删除操作，检查位置附近的上下文
                # 用于匹配语义错误（大模型返回整个短语，但diff只显示插入/删除）
                if change["tag"] in ["insert", "delete"]:
                    # 获取变化位置前后的上下文（前后各5个字符）
                    pos = change["position"]
                    context_start = max(0, pos - 5)
                    context_end = min(len(source), pos + 5)
                    context = source[context_start:context_end]

                    # 检查大模型的错误短语是否出现在这个上下文中
                    if detail_phrase in context or context in detail_phrase:
                        matched_detail = detail
                        used_details.add(idx)
                        break

            # 构建错误信息
            if matched_detail:
                # 有大模型详情：使用大模型的类型和解释
                error_type = matched_detail.error_type
                explanation = matched_detail.explanation

                # 语义错误：不显示修正建议
                if error_type == "semantic":
                    errors.append(
                        {
                            "original": change["original"],
                            "corrected": "",
                            "position": change["position"],
                            "error_type": error_type,
                            "explanation": explanation,
                        }
                    )
                else:
                    # 其他错误：显示修正
                    errors.append(
                        {
                            "original": change["original"],
                            "corrected": change["corrected"],
                            "position": change["position"],
                            "error_type": error_type,
                            "explanation": explanation,
                        }
                    )
            else:
                # 没有匹配到大模型详情：使用 diff 的默认类型
                if change["tag"] == "replace":
                    error_type = "typo"
                    explanation = ""
                elif change["tag"] == "delete":
                    error_type = "redundant"
                    explanation = "多余的内容"
                elif change["tag"] == "insert":
                    error_type = "missing"
                    explanation = "缺少的内容"
                else:
                    error_type = "unknown"
                    explanation = ""

                errors.append(
                    {
                        "original": change["original"],
                        "corrected": change["corrected"],
                        "position": change["position"],
                        "error_type": error_type,
                        "explanation": explanation,
                    }
                )

        return errors

    def correct_text(self, text: str) -> Dict[str, Any]:
        """
        纠正单个文本

        Args:
            text: 待纠正的文本

        Returns:
            {
                "source": "原始文本",
                "target": "修正后的文本",
                "errors": [
                    {
                        "original": "错误内容",
                        "corrected": "正确内容",
                        "position": 位置（精确），
                        "error_type": "错误类型",
                        "explanation": "错误原因"
                    }
                ]
            }
        """
        try:
            # 1. 构建 prompt
            prompt = self.prompt_template.format(text=text)

            # 2. 调用大模型
            logger.info(f"正在调用千问 {self.model} 检查文本...")
            response = self.llm.invoke(prompt)

            # 3. 解析结构化输出（response 是 AIMessage，需要取 .content）
            logger.info(f"千问响应: {response.content}")
            parsed_output: QwenCorrectionOutput = self.output_parser.parse(
                response.content
            )

            # 4. 使用 diff 算法精确计算位置
            errors = self._calculate_precise_positions(
                source=text,
                target=parsed_output.corrected_text,
                error_details=parsed_output.error_details,
            )

            # 5. 返回统一格式
            return {
                "source": text,
                "target": parsed_output.corrected_text,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"千问纠错失败: {e}")
            # 返回原文，表示未检测到错误
            return {"source": text, "target": text, "errors": []}

    def correct_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量纠正文本（并发处理）"""
        # 使用线程池并发调用 API
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.correct_text, texts))
        return results

    async def correct_texts_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量纠正文本（异步并发）"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.correct_text, text)
            for text in texts
        ]
        results = await asyncio.gather(*tasks)
        return list(results)
