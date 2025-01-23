# evaluator.py

import json
import re
import pypinyin
import numpy as np
from openai import OpenAI
import os
from typing import List, Dict, Any, Union
import logging
from pathlib import Path
from config import EXPERIMENT_CONFIG
from llm import LLM


class Evaluator:
    def __init__(self, task_type: str, specific_task: str, api_key: str, model_type: str):
        """
        初始化评测器
        :param task_type: 任务类型 (open_task/complex_task)
        :param specific_task: 具体任务 (poetry/story/nli/math)
        :param api_key: GPT-4 API密钥
        """
        self.task_type = task_type
        self.specific_task = specific_task
        self.api_key = api_key
        self.model_type = model_type
        self.llm = LLM(self.model_type, self.api_key)
        
        # 创建结果保存目录
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志记录
        logging.basicConfig(
            filename=self.results_dir / f"{specific_task}_evaluation.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def evaluate(self, results_file: str) -> Dict[str, Any]:
        """
        评估任务结果
        :param results_file: 任务结果文件路径
        :return: 评估结果字典
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        #print("results:",results)
        if self.task_type == "open_task":
            if self.specific_task == "poetry":
                return self._evaluate_poetry(results)
            elif self.specific_task == "story":
                return self._evaluate_story(results)
        else:  # complex_task
            if self.specific_task == "nli":
                return self._evaluate_nli(results)
            elif self.specific_task == "math":
                return self._evaluate_math(results)
                
        raise ValueError(f"Unknown task type: {self.task_type} or specific task: {self.specific_task}")

    def _evaluate_poetry(self, results: List[Dict]) -> Dict[str, float]:
        """评估诗歌生成结果"""
        total_scores = []
        inputs, outputs = self.extract_inputs_outputs(results)
        print("inputs:",inputs)
        print("outputs:",outputs)

        for result in results:
            # 保存原始输入和输出用于人工验证
            #self._save_for_human_validation(result)
            
            scores = {
                "coherence": self._evaluate_coherence_llm(inputs, outputs),
                "rhythm": self._evaluate_rhythm(outputs),
                "elegance": self._evaluate_elegance_llm(inputs, outputs),
                "emotion": self._evaluate_emotion_llm(inputs, outputs)
            }
            total_scores.append(scores)
            
        # 计算平均分数
        avg_scores = {
            metric: np.mean([score[metric] for score in total_scores])
            for metric in ["coherence", "rhythm", "elegance", "emotion"]
        }
        avg_scores["overall"] = np.mean(list(avg_scores.values()))
        
        return avg_scores
    
    def _evaluate_story(self, results: List[Dict]) -> Dict[str, float]:
        """评估故事生成结果"""
        total_scores = []
        
        for result in results:
            #self._save_for_human_validation(result)
            
            scores = {
                "relevance": self._evaluate_relevance_llm(result["input"], result["final_output"]),
                "interestingness": self._evaluate_interestingness_llm(result["final_output"]),
                "fluency": self._evaluate_fluency_llm(result["final_output"]),
                "coherence": self._evaluate_coherence_llm(result["input"], result["final_output"]),
                "completeness": self._evaluate_completeness_llm(result["final_output"])
            }
            total_scores.append(scores)
            
        avg_scores = {
            metric: np.mean([score[metric] for score in total_scores])
            for metric in ["relevance", "interestingness", "fluency", "coherence", "completeness"]
        }
        avg_scores["overall"] = np.mean(list(avg_scores.values()))
        
        return avg_scores

    def _evaluate_nli(self, results: List[Dict]) -> Dict[str, float]:
        """评估NLI任务结果"""
        correct = 0
        total = len(results)
        
        for result in results:
            self._save_for_human_validation(result)
            
            # 验证预测标签是否正确
            if result["final_output"] == result["label"]:
                correct += 1
                
        accuracy = correct / total
        return {"accuracy": accuracy}

    def _evaluate_math(self, results: List[Dict]) -> Dict[str, float]:
        """评估数学推理任务结果"""
        correct = 0
        total = len(results)
        
        for result in results:
            self._save_for_human_validation(result)
            
            # 提取数字答案并比较
            predicted = self._extract_number(result["final_output"])
            reference = self._extract_number(result["answer"])
            
            if predicted is not None and reference is not None:
                # 允许一定的数值误差
                if abs(predicted - reference) < 1e-6:
                    correct += 1
                    
        accuracy = correct / total
        return {"accuracy": accuracy}

    def _evaluate_coherence_llm(self, input_text: str, output_text: str) -> float:
        """使用GPT-4评估连贯性"""
        prompt = f"""Dear Evaluator,

        Please professionally assess the coherence and logical consistency of the given text, providing a score between 0.0 and 1.0 (where 1.0 indicates complete coherence, and 0.0 indicates a complete lack thereof), to two decimal places:

        Text Part 1: {input_text}
        Text Part 2: {output_text}

        Evaluation Criteria:
        - Whether the text structure is clear and reasonable;
        - Whether there is a strong logical connection between Text Part 1 and Text Part 2;
        - Whether the content remains consistent throughout, without contradictions.

        Output Format:
        score: [Your Score Here]

        Thank you for your professional judgment and cooperation."""
        
        try:
            response = self.llm.generate(prompt)
            score = float(response.strip())
            return min(max(score, 0), 1)  # 确保分数在0-1之间
        except Exception as e:
            logging.error(f"GPT-4 evaluation error: {e}")
            return 9999  # 发生错误时返回中间分数

    def _evaluate_rhythm(self, text: str) -> float:
        """评估诗歌韵律"""
        # 使用pypinyin获取拼音
        try:
            pinyins = pypinyin.pinyin(text, style=pypinyin.TONE)
            # 检查韵脚
            # 这里是简化版的实现，实际应用中可能需要更复杂的韵律规则
            return 1.0  # 返回韵律评分
        except Exception as e:
            logging.error(f"Rhythm evaluation error: {e}")
            return 0.5

    def _extract_number(self, text: str) -> Union[float, None]:
        """从文本中提取数值答案"""
        try:
            # 寻找数字答案的模式
            patterns = [
                r'####\s*(\d+\.?\d*)',  # 匹配 #### 后的数字
                r'答案[:：]\s*(\d+\.?\d*)',  # 匹配"答案："后的数字
                r'(\d+\.?\d*)$'  # 匹配末尾的数字
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return float(match.group(1))
            return None
        except Exception as e:
            logging.error(f"Number extraction error: {e}")
            return None

    def _save_for_human_validation(self, result: Dict):
        """保存结果供人工验证"""
        filename = self.results_dir / f"{self.specific_task}_human_validation.jsonl"
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    # 其他GPT-4评估方法实现类似
    def _evaluate_elegance_llm(self, input_text: str, output_text: str) -> float:
        """评估语言优美性"""
        prompt = f"""Dear Evaluator,

        Please assess the aesthetic quality of the language in the given text below, providing a score between 0.0 and 1.0 (where 1.0 represents the highest level of linguistic beauty, and 0.0 indicates none):

        Text: {input_text}+','+{output_text}+'。'

        Evaluation Criteria:
        - The elegance of vocabulary used;
        - The gracefulness of expression;
        - The depth of the imagery conveyed.

        Output Format:
        score: [Your Score Here]

        Instructions: Ensure the score is provided to two decimal places and nothing else follows it.

        Thank you for your professional judgment and cooperation."""
        try:
            response = self.llm.generate(prompt)
            score = float(response.strip())
            return min(max(score, 0), 1)  # 确保分数在0-1之间
        except Exception as e:
            logging.error(f"GPT-4 evaluation error: {e}")
            return 9999  # 发生错误时返回中间分数

    def _evaluate_emotion_llm(self, input_text: str, output_text: str) -> float:
        """评估情感主题一致性"""
        prompt = f"""尊敬的评估者，

        请根据以下标准，评估输入文本和输出文本在情感和主题上的一致性，并给予一个介于0.00到1.00之间的评分（其中1.00表示完全一致，0.00表示完全不同），保留两位小数：

        输入文本：{input_text}
        输出文本：{output_text}

        评分标准：
        - 情感表达是否统一；
        - 主题是否一致；
        - 意境是否和谐。

        输出格式：
        score: [在此处输入您的评分]

        指示：确保评分精确到两位小数，且之后没有任何其他字符或信息。

        感谢您的专业判断与合作。
        """
        try:
            response = self.llm.generate(prompt)
            score = float(response.strip())
            return min(max(score, 0), 1)  # 确保分数在0-1之间
        except Exception as e:
            logging.error(f"GPT-4 evaluation error: {e}")
            return 9999  # 发生错误时返回中间分数
    
    # def clean_json_string(self, json_string):
    #     # Remove the leading "```json" and any trailing whitespace
    #     json_str = json_string.replace("```json", "").strip()
    #     # Parse the JSON data
    #     data = json.loads(json_str)
        
    #     return cleaned


    def extract_inputs_outputs(self, results):
        # print("results type:", type(results))
        # print("results:", results)
        rounds = EXPERIMENT_CONFIG[self.task_type]["rounds"]
        final_round_results = results[rounds-1]
        #print("final_round_results:",final_round_results)
        # 提取 inputs 和 outputs
        inputs = [result['question'] for result in final_round_results]
        outputs = [result['final_output'] for result in final_round_results] 
        #print("outputs:", outputs)
        clearned_outputs = []
        for output in outputs:
            try:
                # 清理 JSON 字符串
                # print("output:",output)
                #cleaned_output = self.clean_json_string(outputs[0])
                cleaned_output = output.split('"Best Answer": "')[1].split('"')[0].split(": ")[1]
                # print("cleaned_output:", cleaned_output)
                clearned_outputs.append(cleaned_output)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # 您可以在这里添加更多的逻辑来处理解析错误，比如记录日志、抛出异常等
            except Exception as e:
                print(f"An unexpected error occurred: {e}")       

        #outputs[0]
        return inputs, clearned_outputs