# evaluation.py

import openai
import os
import json
# from pypinyin import lazy_pinyin, Style
from typing import List, Dict
from datetime import datetime
import logging
import openai
from openai import OpenAI
from llm import LLM
# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class Evaluation:
    def __init__(self, output_dir: str = "evaluation_results", model_type: str = None, api_key: str = None):
        """
        初始化评测模块
        :param output_dir: 评测结果保存目录
        :param gpt_api_key: GPT-4 API Key
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_type = model_type
        self.api_key = api_key
        self.llm = LLM(self.model_type, self.api_key)

    def evaluate_poetry(self, outputs: List[str], references: List[str]) -> Dict[str, float]:
        """
        评测诗歌生成任务
        :param outputs: 模型生成的诗句列表
        :param references: 参考答案的诗句列表
        :return: 各指标的分数
        """
        scores = {
            "Coherence": 0.0,  # 连贯性
            "Rhythm": 0.0,     # 韵律与节奏
            "Elegance": 0.0,   # 语言优美性
            "Emotion": 0.0     # 情感与主题一致性
        }
        total = len(outputs)

        for idx, (output, reference) in enumerate(zip(outputs, references)):
            # Coherence, Elegance, Emotion: 使用 GPT-4 自动评分
            gpt_scores = self._score_poetry(output, reference)
            scores["Coherence"] += gpt_scores["Coherence"]
            scores["Elegance"] += gpt_scores["Elegance"]
            scores["Emotion"] += gpt_scores["Emotion"]

            # Rhythm: 使用 pypinyin 检测韵脚
            scores["Rhythm"] += self._check_rhythm(output, reference)

        # 计算平均分
        scores_avg = {k: v / total for k, v in scores.items()}

        # 保存结果
        #self._save_results("poetry", outputs, references, scores["Coherence"], scores["Elegance"], scores["Emotion"], scores["Rhythm"], scores_avg)
        return scores,scores_avg

    def evaluate_story(self, outputs: List[str], references: List[str], keywords: List[List[str]]) -> Dict[str, float]:
        """
        评测故事生成任务
        :param outputs: 模型生成的故事列表
        :param references: 参考答案的故事列表
        :param keywords: 每个故事的关键词列表
        :return: 各指标的分数
        """
        scores = {
            "Relevance": 0.0,          # 相关性
            "Interestingness": 0.0,    # 趣味性
            "Fluency": 0.0,            # 流畅性
            "Coherence": 0.0,          # 连贯性
            "Completeness": 0.0        # 完整性
        }
        total = len(outputs)

        for idx, (output, reference, keyword_set) in enumerate(zip(outputs, references, keywords)):
            # Relevance, Interestingness, Fluency, Coherence, Completeness: 使用 GPT-4 自动评分
            gpt_scores = self._gpt4_score_story(output, reference, keyword_set)
            scores["Relevance"] += gpt_scores["Relevance"]
            scores["Interestingness"] += gpt_scores["Interestingness"]
            scores["Fluency"] += gpt_scores["Fluency"]
            scores["Coherence"] += gpt_scores["Coherence"]
            scores["Completeness"] += gpt_scores["Completeness"]

        # 计算平均分
        scores = {k: v / total for k, v in scores.items()}

        # 保存结果
        self._save_results("story", outputs, references, scores)
        return scores

    def _score_poetry(self, output: str, reference: str) -> Dict[str, float]:
        """
        使用 GPT-4 对诗歌生成任务进行评分
        :param output: 模型生成的诗句
        :param reference: 参考答案的诗句
        :return: 各指标的分数
        """
        prompt = f"""
        以下是一句古诗和模型生成的下一句，请根据以下标准对生成的诗句进行评分：
        1. **连贯性（Coherence）**：生成的诗句是否与上一句逻辑连贯，意境一致。（0到1分）
        2. **语言优美性（Elegance）**：生成的诗句是否具有优美的语言表达。（0到1分）
        3. **情感与主题一致性（Emotion）**：生成的诗句是否体现了与上一句一致的情感和主题。（0到1分）

        上一句古诗："{reference}"
        生成的下一句："{output}"

        请按以下格式返回：
        Coherence: X.X
        Elegance: X.X
        Emotion: X.X
        """
        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _score_story(self, output: str, reference: str, keywords: List[str]) -> Dict[str, float]:
        """
        使用 GPT-4 对故事生成任务进行评分
        :param output: 模型生成的故事
        :param reference: 参考答案的故事
        :param keywords: 关键词列表
        :return: 各指标的分数
        """
        prompt = f"""
        以下是一组关键词、一个参考故事，以及模型生成的故事。请根据以下标准对生成的故事进行评分：
        1. **相关性（Relevance）**：生成的故事是否很好地包含了所有关键词。（0到1分）
        2. **趣味性（Interestingness）**：故事是否有趣且吸引人。（0到1分）
        3. **流畅性（Fluency）**：单句是否语法正确且流畅。（0到1分）
        4. **连贯性（Coherence）**：故事整体是否逻辑连贯。（0到1分）
        5. **完整性（Completeness）**：故事是否有完整的开头、发展和结尾。（0到1分）

        关键词：{', '.join(keywords)}
        参考故事："{reference}"
        生成的故事："{output}"

        请按以下格式返回：
        Relevance: X.X
        Interestingness: X.X
        Fluency: X.X
        Coherence: X.X
        Completeness: X.X
        """
        response = self._call_gpt4(prompt)
        return self._parse_gpt4_response(response)

    def _parse_response(self, response: str) -> Dict[str, float]:
        """
        解析 GPT-4 的评分结果
        :param response: GPT-4 返回的文本
        :return: 各评分项
        """
        scores = {}
        for line in response.splitlines():
            if ":" in line:
                key, value = line.split(":")
                key = key.strip()
                value = float(value.strip())
                scores[key] = value
        return scores