# main.py

import re
import time
import pandas as pd
import random
import openai
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from agent import Agent
from task_executor import TaskExecutor
from logger import Logger
from data_loader import DataLoader
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from config import EXPERIMENT_CONFIG
from typing import List, Dict, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pypinyin
import logging
import numpy as np

import sys
import io
import ast

# 设置默认编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables from local/project configuration.
_ = load_dotenv(find_dotenv())

# Add your API key here, if needed
openai.api_key = os.environ.get("OPENAI_API_KEY")
deep_seek_api_key = os.environ.get("DEEPSEEK_API_KEY")

class LLM:
    def __init__(self, model_type, api_keys, model):
        """
        Initialize the agent.
        :param personality: MBTI personality type
        :param model_type: Model type (e.g., gpt4, deepseek, qwen2.5)
        :param api_keys: Dictionary containing API keys for the models
        """
        
        self.model_type = model_type
        self.api_keys = api_keys
        self.model = model

    def generate(self, prompt, max_tokens=1024, temperature=0.7, response_format="text", system_prompt:str="You are a helpful assistant"):
        """
        Generate a response based on the model type.
        :param prompt: Input prompt
        :param max_tokens: Maximum tokens for the response
        :param temperature: Controls randomness of the output
        :return: Generated text
        """
        if self.model_type == "gpt4":
            return self._generate_gpt4(prompt, max_tokens, temperature)
        elif self.model_type == "deepseek":
            return self._generate_deepseek(prompt, max_tokens, temperature, response_format, system_prompt)
        elif self.model_type == "qwen2.5":
            return self._generate_qwen(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def _generate_gpt4(self, prompt, max_tokens, temperature):
        """
        Call the Deepseek API.
        """
        openai.api_key = self.api_keys 
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  # or "Deepseek"
                prompt=self._add_personality_context(prompt),
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating response from Deepseek: {e}")
            return None

    # def _generate_deepseek(self, prompt, max_tokens, temperature, response_format:str="text"):
    #     """
    #     Call the DeepSeek API.
    #     """
    #     try:
    #         client = OpenAI(api_key=self.api_keys, base_url="https://api.deepseek.com")
    #         response = client.chat.completions.create(
    #         model = self.model,   # "deepseek-chat"
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant"},
    #             {"role": "user", "content": prompt},
    #         ],
    #         response_format={"type": response_format},
    #         max_tokens = max_tokens,
    #         temperature = temperature,
    #         stream=False
    #     )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         if "429" in str(e):  # 检测速率限制错误
    #             print(f"Rate limit reached. Retrying in 10 seconds...")
    #             time.sleep(10)  # 等待 10 秒后重试
    #             return self._generate_deepseek(prompt, max_tokens, temperature, response_format)
    #         else:
    #             print(f"Error generating response from DeepSeek: {repr(e)}")  # 使用 repr 确保正确显示非 ASCII 字符
    #             return None

    def _generate_deepseek(self, prompt, max_tokens, temperature, response_format:str="text", system_prompt:str="You are a helpful assistant"):
        """
        Call the DeepSeek API from siliconflow.
        max_tokens: default:512, Required range: 1 < x < 8192
        temperature: default: 0.7
        top_k: default:50
        response_format: Possible values: [text, json_object]
        """
        try:
            client = OpenAI(api_key=self.api_keys, base_url="https://api.siliconflow.cn/v1")
            response = client.chat.completions.create(
                model = self.model,   # "deepseek-ai/DeepSeek-V3"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                response_format={"type": response_format},
                max_tokens = max_tokens,
                temperature = temperature,
                stream=False
            )
            # for chunk in response:
            #     print(chunk.choices[0].delta.content, end='')
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):  # 检测速率限制错误
                print(f"Rate limit reached. Retrying in 10 seconds...")
                time.sleep(10)  # 等待 10 秒后重试
                return self._generate_deepseek(prompt, max_tokens, temperature, response_format, system_prompt)
            else:
                print(f"Error generating response from DeepSeek: {e}")  # 确保在 except 块内引用 e
                return None

    # def _generate_deepseek(self, prompt, max_tokens, temperature, response_format:str="text"):
    #     """
    #     Call the DeepSeek API from aliyuncs.
    #     max_tokens: default:512, Required range: 1 < x < 8192
    #     temperature: default: 0.7
    #     top_k: default:50
    #     response_format: Possible values: [text, json_object]
    #     """
    #     try:
    #         client = OpenAI(api_key=self.api_keys, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    #         response = client.chat.completions.create(
    #             model = self.model,   # "deepseek-ai/DeepSeek-V3"
    #             messages=[
    #                 {'role': 'user', 
    #                 'content': prompt}
    #             ],
    #             response_format={"type": response_format},
    #             max_tokens = max_tokens,
    #             temperature = temperature,
    #             stream=False
    #         )
    #         # for chunk in response:
    #         #     print(chunk.choices[0].delta.content, end='')
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         if "429" in str(e):  # 检测速率限制错误
    #             print(f"Rate limit reached. Retrying in 10 seconds...")
    #             time.sleep(10)  # 等待 10 秒后重试
    #             return self._generate_deepseek(prompt, max_tokens, temperature, response_format)
    #         else:
    #             print(f"Error generating response from DeepSeek: {e}")  # 确保在 except 块内引用 e
    #             return None
        

    def _generate_qwen(self, prompt, max_tokens, temperature):
        """
        Call the Qwen-2.5 API.
        """
        try:
            headers = {'Content-Type': 'application/json'}
            data = {"prompt": prompt}
            response = requests.post(url='http://127.0.0.1:5008', headers=headers, data=json.dumps(data))
            return response.json()['response']
        except Exception as e:
            print(f"Error generating response from Qwen-2.5: {e}")
            return None


def _generate_prompt(data_item, task_type, specific_task):
    """
    Generate prompt according to task type
    :param data_item: Current data item
    :param round_num: Current round
    :return: prompt 
    """
    # print("data_item:",data_item)
    if task_type == "open_task":
        if specific_task == "poetry":  # Poetry Generation
            return (
f"""
You are a highly talented Chinese ancient poet, renowned for composing exquisite Chinese poetry. I will provide the opening line of a poem, and your task is to craft the next line based on its artistic conception, rhythm, and style, creating a harmonious and beautiful continuation.
    
Please adhere to the following requirements:
1. **Originality**: Your line must be entirely original, not borrowed from any existing poetry.
2. **Rhythmic Consistency**: Ensure the tonal pattern and rhyme scheme align perfectly with the previous line.
3. **Artistic Continuity**: Your line should seamlessly extend the meaning, atmosphere, and imagery of the given line, evoking a sense of beauty in nature, emotion, or philosophy.
4. **Elegance in Language**: Employ refined and elegant diction that reflects the style and artistry of classical Chinese poetry.
5. **Strict Output Format**: Only output a single poetic line, without any additional explanations or context.

**Opening Line:** {data_item}
**Next Line:**
"""
            )
        elif specific_task == "story":  # Story Generation
            return (
                "You are an imaginative and skilled storyteller, renowned for crafting captivating and engaging narratives. I will provide you with a set of keywords, and your task is to write a short story that seamlessly incorporates these keywords into its plot."
                "\n\nPlease adhere to the following requirements:\n"
                "1. **Incorporate All Keywords**: The story must include all the provided keywords, integrating them naturally and meaningfully into the storyline.\n"
                "2. **Structured Narrative**: Ensure the story has a clear beginning, middle, and end, with a logical and coherent progression.\n"
                "3. **Engaging and Vivid Language**: Use descriptive and vivid language to bring the story to life and captivate the audience.\n"
                "4. **Creativity**: Demonstrate originality and creativity, making the story unique and memorable.\n\n"
                f"**Keywords**: {', '.join(data_item)}\n**Start your story:**"
            )

    elif task_type == "complex_task":
        if specific_task == "nli":  # NLI
            return (
                "You are a logical reasoning assistant with exceptional skills in analyzing the relationship between a premise and a hypothesis. I will provide you with a pair of sentences, and your task is to determine their relationship and explain your reasoning."
                "\n\nPlease adhere to the following requirements:\n"
                "1. **Relationship Determination**: Identify whether the relationship is 'entailment', 'neutral', or 'contradiction', and map it to the corresponding label value:\n"
                "   - 'entailment' → label value: 0\n"
                "   - 'neutral' → label value: 1\n"
                "   - 'contradiction' → label value: 2\n"
                "2. **Output Format**: Strictly output only the label value and a concise explanation of your reasoning.\n"
                "   Example:\n"
                "   - 0: The hypothesis logically follows from the premise.\n"
                "   - 1: The hypothesis is unrelated or lacks sufficient information to infer from the premise.\n"
                "   - 2: The hypothesis contradicts the premise.\n\n"
                f"**Premise**: {data_item['premise']}\n"
                f"**Hypothesis**: {data_item['hypothesis']}\n\n"
                "**Start your analysis and provide your output:**"
            )
        elif specific_task == "math":  # Mathematical Reasoning
            # question = data_item["question"]
            return (
f"""
Can you solve the following math problem?
{data_item['question']}

Your final answer should be a single numerical number and explain your reasoning.

Please adhere to the following JSON OUTPUT format:
{{
    "answer":"<final answer should be a single numerical number>",
    "Explanation": "<Reasons for the answer>"
}}

"""
            )

    raise ValueError(f"Unsupported task type: {specific_task}")
    
def _prepare_initial_prompts(data, task_type, specific_task):
    """
    Generate initial prompt according to task type
    :return: Initial prompt list
    """
    #print("self.data:",self.data)
    #data = data[:3]

    prompts = []
    for data_item in data:
        if specific_task == "poetry":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
            # print("prompts:",prompts)
        elif specific_task == "story":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
        elif specific_task == "nli":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
        elif specific_task == "math":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
        else:
            raise ValueError(f"Unsupported specific task type: {specific_task}")
    
    return prompts


def _run_round(inputs, agent, max_tokens=1024, temperature=0.7, response_format="text", system_prompt:str="You are a helpful assistant"):
    question_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # 根据API限制调整并发数
        futures = []
        for original_question, prompt in inputs:
            future = executor.submit(agent.generate, prompt, max_tokens, temperature, response_format, system_prompt)
            futures.append((original_question, prompt, future))
        
        for original_question, prompt, future in futures:
            output = future.result()
            question_results.append({
                "question": original_question,
                "input": prompt,
                "output": output,
            })
    return question_results

class Evaluator:
    def __init__(self, task_type: str, specific_task: str, api_key: str, model_type: str, model: str):
        """
        初始化评测器
        :param task_type: 任务类型 (open_task/complex_task)
        :param specific_task: 具体任务 (poetry/story/nli/math)
        :param api_key: Deepseek API密钥
        """
        self.task_type = task_type
        self.specific_task = specific_task
        self.api_key = api_key
        self.model_type = model_type
        self.model = model
        self.llm = LLM(self.model_type, self.api_key, self.model)
        
        # 创建结果保存目录
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志记录
        logging.basicConfig(
            filename=self.results_dir / f"{specific_task}_exp1_no-mbti-mac_evaluation.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def evaluate(self, results):
        """
        评估任务结果
        :param results_file: 任务结果文件路径
        :return: 评估结果字典
        """
        
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
    
    # def parse_json_from_markdown(self, s):
    #     # 匹配 ```json 和 ``` 之间的内容（支持跨行）
    #     match = re.search(r'```json(.*?)```', s, re.DOTALL)
    #     if match:
    #         json_str = match.group(1).strip()
    #         try:
    #             return json.loads(json_str)
    #         except json.JSONDecodeError as e:
    #             raise ValueError("Invalid JSON format") from e
    #     raise ValueError("No JSON code block found")


    def _evaluate_poetry(self, results: List[Dict]):
        """评估诗歌生成结果"""
        total_scores = []
        #inputs, outputs = self.extract_inputs_outputs(results)
        # print("inputs:",inputs)
        # print("outputs:",outputs)
        

        for result in results:
            # 保存原始输入和输出用于人工验证
            #self._save_for_human_validation(result)
            prompt = f"""
            You are an AI reviewer with expertise in Chinese classical poetry. Your task is to evaluate poetic sentences generated in a continuation task. 
            Based on the given first half of a poem: "{result['question']}" , and the generated second half: "{result['output']}", you will assign scores (range: 0-1.00, rounded to two decimal places) across multiple dimensions and provide detailed feedback for each dimension. The final output should be in **JSON format**.

            ### **Evaluation Dimensions**:
            1. **Semantic Coherence**  
            - Does the second half maintain semantic coherence with the first half? Is the expression complete, and does the language flow naturally?

            2. **Rhythm and Harmony**  
            - Is the rhythm of the second half harmonious and smooth? Does it create an overall sense of musicality with the first half, considering sound and cadence (without being strictly constrained by traditional tonal and rhyming rules)?

            3. **Imagery and Aesthetic Appeal**  
            - Does the second half exhibit beautiful imagery? Does it evoke a vivid sense of aesthetics and artistic expression?

            4. **Emotional Depth**  
            - Does the second half deepen the emotional expression? Does it further develop or refine the sentiment introduced in the first half?

            ### **JSON Output Format**:
            Your results and feedback should be structured as follows:

            ```json
            {{
                "semantic_coherence": {{
                    "score": 0.00,
                    "comment": "Feedback on semantic coherence"
                }},
                "rhythm_and_harmony": {{
                    "score": 0.00,
                    "comment": "Feedback on rhythm and harmony"
                }},
                "imagery_and_aesthetic": {{
                    "score": 0.00,
                    "comment": "Feedback on imagery and aesthetic appeal"
                }},
                "emotional_depth": {{
                    "score": 0.00,
                    "comment": "Feedback on emotional depth"
                }}
            }}
            """
            scores = self.multi_response(prompt, max_attempts=5, max_tokens=1024, temperature=0.7, response_format="json_object")
            total_scores.append(scores)
            
        return total_scores
    
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

    def _evaluate_nli(self, results: List[Dict]):
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



    # def _evaluate_math(self, results: List[Dict]):
    #     """评估数学推理任务结果"""
    #     correct = 0
    #     total = len(results)
    #     for index, row in results.iterrows():
    #         try:
    #             reference = ast.literal_eval(row["question"])  #字符串转换成字典
    #             reference_ans = float(reference["answer"])
    #             predicted = json.loads(row["output"])  #字符串转换成字典
    #             #predicted = ast.literal_eval(row["output"])  #字符串转换成字典
    #             predicted_ans = float(predicted["answer"])

    #             if predicted_ans is not None and reference_ans is not None:
    #                 # 允许一定的数值误差
    #                 if abs(predicted_ans - reference_ans) < 1e-6:
    #                     correct += 1
    #         except (SyntaxError, ValueError) as e:
    #             print(f"Error parsing row {index}")
    #             continue  # 跳过非法行

    #     accuracy = correct / total
    #     return accuracy


    # def repair_output(self, output: str) -> str:
    #     """
    #     修复错误的 JSON 输出数据。
    #     :param output: 错误的 JSON 字符串
    #     :return: 修复后的 JSON 字符串
    #     """
    #     try:
    #         # 修复 Explanation 中的换行
    #         fixed_explanation = output.replace("\n", " ").replace("  ", " ").strip()  # 去掉多余空格和错误换行
    #         fixed_explanation = fixed_explanation.replace("1.", "\n 1.").replace("2.", "\n2.").replace("3.", "\n3.")  # 修复有序列表换行
    #         return fixed_explanation

    #     except Exception as e:
    #         print(f"Failed to repair output: {e}")
    #         return output  # 如果修复失败，返回原始输出作为最后手段

    def convert_value(x):
        if isinstance(x, (int, float)):
            return float(x)
        elif isinstance(x, str):
            try:
                return float(x)
            except ValueError:
                return x
        else:
            return str(x)
    

    def _evaluate_math(self, results: List[Dict]):
        """评估数学推理任务结果"""
        correct = 0
        total = len(results)
        problem_rows = []  # 用于记录问题行的信息
        repaired_rows = []  # 用于记录修复的行
        unrepaired_rows = []  # 记录无法修复的行

        results_save = pd.DataFrame({
            'index':[],
            'reference_ans': [], # 初始化为空或其他默认值
            'predicted_ans': [],  # 初始化为空或其他默认值
            'labels': []  # 初始化为空或其他默认值
        })

        for index, row in results.iterrows():
            try:
                # 尝试解析 question 和 output
                reference = ast.literal_eval(row["question"])  # 转换字符串为字典
                # reference_ans = float(reference["answer"])
                reference_ans = reference["answer"]
                predicted = ast.literal_eval(row["output"])  # 转换字符串为字典
                # predicted_ans = float(predicted["answer"])
                predicted_ans = predicted["answer"]

                # 判断答案是否正确（允许一定的误差）
                # if abs(predicted_ans - reference_ans) < 1e-6:
                #     correct += 1
                if predicted_ans == reference_ans:
                    new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': True}])
                    results_save = pd.concat([results_save, new_row], ignore_index=True)
                    correct += 1
                else:
                    try:
                        if isinstance(predicted_ans, (int, float)) and isinstance(reference_ans, (int, float)):
                            if float(predicted_ans) == float(reference_ans):
                                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': True}])
                                results_save = pd.concat([results_save, new_row], ignore_index=True)
                                correct += 1
                            else:
                                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': False}])
                                results_save = pd.concat([results_save, new_row], ignore_index=True)
                        elif isinstance(predicted_ans, (str)) and isinstance(reference_ans, (str)):
                            if float(predicted_ans) == float(reference_ans):
                                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': True}])
                                results_save = pd.concat([results_save, new_row], ignore_index=True)
                                correct += 1
                            else:
                                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': False}])
                                results_save = pd.concat([results_save, new_row], ignore_index=True)
                        else:
                            # 尝试转换为str进行比较
                            if str(predicted_ans) == str(reference_ans):
                                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': True}])
                                results_save = pd.concat([results_save, new_row], ignore_index=True)
                                correct += 1
                            else:
                                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': False}])
                                results_save = pd.concat([results_save, new_row], ignore_index=True)

                    except (TypeError, ValueError):            
                        new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': False}])
                        results_save = pd.concat([results_save, new_row], ignore_index=True)
            except Exception as e:
                # 捕获异常并记录问题行
                print(f"Error parsing row {index},{e}")
                #problem_rows.append((index, row["output"], str(e)))
                new_row = pd.DataFrame([{'index':index, 'reference_ans': reference, 'predicted_ans': row["output"], 'labels': 9999}])
                results_save = pd.concat([results_save, new_row], ignore_index=True)
                # if "unterminated string literal" in str(e):
                #     o = row["output"].replace("\n", " ").strip("\n")
                #     predicted = ast.literal_eval(o)  # 去掉多余空格和错误换行
                #     predicted_ans = predicted["answer"]
                #     if predicted_ans == reference_ans:
                #         new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': True}])
                #         results_save = pd.concat([results_save, new_row], ignore_index=True)
                #         correct += 1
                #     else:
                #         new_row = pd.DataFrame([{'index':index, 'reference_ans': reference_ans, 'predicted_ans': predicted_ans, 'labels': False}])
                #         results_save = pd.concat([results_save, new_row], ignore_index=True)
                # else:
                #     # 捕获异常并记录问题行
                #     print(f"Error parsing row {index},{e}")
                #     #problem_rows.append((index, row["output"], str(e)))
                #     new_row = pd.DataFrame([{'index':index, 'reference_ans': reference, 'predicted_ans': predicted, 'labels': 9999}])
                #     results_save = pd.concat([results_save, new_row], ignore_index=True)

        # # 尝试修复问题行
        # for index, output, error in problem_rows:
        #     try:
        #         # 修复问题行
        #         repaired_output = self.repair_output(output)
        #         print(f"Repaired row {index}: {repaired_output}")

        #         # 尝试重新解析修复后的字符串
        #         predicted = ast.literal_eval(repaired_output)  # 转换字符串为字典
        #         predicted_ans = float(predicted["answer"])
        #         reference = ast.literal_eval(results.loc[index, "question"])
        #         reference_ans = float(reference["answer"])

        #         # 判断修复后答案是否正确
        #         if abs(predicted_ans - reference_ans) < 1e-6:
        #             correct += 1
        #             repaired_rows.append(index)  # 记录修复成功的行
        #         else:
        #             print(f"Repaired row {index} failed validation.")
        #     except Exception as e:
        #         print(f"Failed to repair row {index}: Error: {e}")
        #         unrepaired_rows.append(index)  # 记录无法修复的行

        # 打印修复结果统计
        accuracy = correct / total
        # print(f"Total rows: {total}")
        # print(f"Correct rows: {correct}")
        # print(f"Problematic rows: {len(problem_rows)}")
        # print(f"Repaired rows: {len(repaired_rows)}")
        # print(f"Unrepaired rows: {len(unrepaired_rows)}")
        return accuracy, results_save
        
    
    def multi_response(self, prompt, max_attempts=5, max_tokens=1024, temperature=0.7, response_format="text"):
        attempt = 0

        while attempt < max_attempts:
            try:
                response = self.llm.generate(prompt, max_tokens, temperature, response_format)
                #return self.parse_json_from_markdown(response)
                return response
            except Exception as e:
                attempt += 1
                if attempt == max_attempts:
                    logging.error(f"Deepseek evaluation error after {max_attempts} attempts: {e}")
                    return 9999  # 所有尝试失败后返回默认分数
                else:
                    logging.error(f"Attempt {attempt} failed, retrying...: {e}")
                    # 可以选择在这里添加一个短暂的等待时间，例如使用time.sleep(1)，避免过快地进行重试
                

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

    
    # def clean_json_string(self, json_string):
    #     # Remove the leading "```json" and any trailing whitespace
    #     json_str = json_string.replace("```json", "").strip()
    #     # Parse the JSON data
    #     data = json.loads(json_str)
        
    #     return cleaned


    def extract_inputs_outputs(self, results):
        
        # 提取 inputs 和 outputs
        inputs = [result['question'] for result in results]
        outputs = [result['output'] for result in results]

        return inputs, outputs
    
            
if __name__ == "__main__":
    # 初始化日志记录器
    Logger.configure_logger("experiment.log")

    # 实验配置
    task_type = "complex_task"  # "open_task" 或 "complex_task"
    specific_task = "math"  # 具体任务类型：poetry, story, nli, math
    model_type = "deepseek"
    model = "deepseek-ai/DeepSeek-V2.5" # "deepseek-v3" # "deepseek-ai/DeepSeek-R1"\"deepseek-ai/DeepSeek-V3"  #deepseek-chat 模型已经升级为 DeepSeek-V3；deepseek-reasoner 模型为新模型 DeepSeek-R1
    data_path = "./data/GSM8K/test.jsonl"  # 数据文件路径  "./data/poetry/poetry.json"  "./data/GSM8K/test.jsonl"
    
    # API 密钥
    #api_keys = os.environ.get("DEEPSEEK_API_KEY")
    api_keys = os.environ.get("DEEPSEEK_API_KEY_GJLD2")
    #api_keys = os.environ.get("DEEPSEEK_API_KEY_aliyuncs")

    # ------------------------------------------------------
    
    # data = DataLoader.load_data(data_path, specific_task)
    # #data = data[:1]
    # #print("data:",data[:2])

    # agent = LLM(model_type = model_type, api_keys = api_keys, model=model)
    # inputs = _prepare_initial_prompts(data, task_type, specific_task)
    # #print("inputs:", inputs)
    # #MBTI_type = "ISTJ"
    # with open("config/mbti_prompts.json", "r", encoding="utf-8") as f:
    #     personality_prompts = json.load(f)
    # # print("personality_prompts:",personality_prompts)
    # # 遍历字典中的每个键值对
    # for index, (key, value) in enumerate(personality_prompts.items()):
    #     if index == 0:
    #         continue  # 跳过第一次循环
    #     MBTI_type = key
    #     MBTI_system_prompt = value
  
    #     # MBTI_system_prompt = personality_prompts.get(MBTI_type, "")
    #     # print("MBTI_system_prompt:", MBTI_system_prompt)
    #     question_results = _run_round(inputs, agent, max_tokens=1024, temperature=0.0, response_format="json_object", system_prompt = MBTI_system_prompt)  #poetry/story:1.5; NLI/math:0.0

    #     #print("question_results:", question_results)
    #     # 定义CSV文件名
    #     filename = f'outputs/exp1_no-mac_{MBTI_type}_gsm8k_DS-v25.csv'
    #     # 将数据转换为 DataFrame
    #     df = pd.DataFrame(question_results)
    #     # 写入 CSV 文件
    #     df.to_csv(filename, index=False)
    #     print(f"Data has been written to {filename}")

    # ---------------------------------------------------------------------------------------
    with open("config/mbti_prompts.json", "r", encoding="utf-8") as f:
        personality_prompts = json.load(f)
    for index, (key, value) in enumerate(personality_prompts.items()):
        MBTI_type = "ESTJ"#key
        filename = f'outputs/GSM8K/exp1_no-mac_{MBTI_type}_gsm8k_DS-v25.csv'
        
        model = "deepseek-ai/DeepSeek-V2.5" # "deepseek-ai/DeepSeek-R1"\"deepseek-ai/DeepSeek-V3"  #deepseek-chat 模型已经升级为 DeepSeek-V3；deepseek-reasoner 模型为新模型 DeepSeek-R1
        question_results = pd.read_csv(filename)
        # question_results = question_results[:3]
        # print("question_results:",question_results)
        
        # 将 DataFrame 转换为字典列表，恢复 question_results 变量
        #question_results = question_results.to_dict(orient='records')

        evaluator = Evaluator(task_type, specific_task, api_keys, model_type, model)
        accuracy, results_save = evaluator.evaluate(question_results)
        print("accuracy:",accuracy)
        results_save.to_csv(f'outputs/GSM8K/exp1_eval_no-mac_{MBTI_type}_gsm8k_DS-v25.csv',index=None)
        print(f"Data has been written to {filename}")
        break

    
    # filename = f'outputs/exp1_no-mac-mbti_gsm8k_DS-v25.csv'
    # model = "deepseek-ai/DeepSeek-V2.5" # "deepseek-ai/DeepSeek-R1"\"deepseek-ai/DeepSeek-V3"  #deepseek-chat 模型已经升级为 DeepSeek-V3；deepseek-reasoner 模型为新模型 DeepSeek-R1
    # question_results = pd.read_csv(filename)
    # evaluator = Evaluator(task_type, specific_task, api_keys, model_type, model)
    # accuracy, results_save = evaluator.evaluate(question_results)
    # print("accuracy:",accuracy)
    # results_save.to_csv(f'outputs/exp1_eval_no-mac_mbti_gsm8k_DS-v25.csv',index=None)
    # print(f"Data has been written to {filename}")



